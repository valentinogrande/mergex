use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mergex::Mergex;
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

fn adder() -> Mergex<i64> {
    Mergex::new(0, 0, |father: &mut i64, son: &i64| *father += *son)
}

/// root with `width` sons, each already set (so every son is dirty)
fn wide(width: usize, dirty: bool) -> Mergex<i64> {
    let root = adder();
    for i in 0..width {
        let son = root.copy();
        if dirty {
            son.set(i as i64);
        }
    }
    root
}

/// a chain root -> son -> ... of `depth` links, deepest node set
fn deep(depth: usize) -> (Mergex<i64>, Mergex<i64>) {
    let root = adder();
    let mut cur = root.clone();
    for _ in 0..depth {
        cur = cur.copy();
    }
    cur.set(1);
    (root, cur)
}

/// copy(): allocating a node and registering it under the sons lock.
/// Timed as "build a tree of `width` sons", divided out by Throughput, with the
/// teardown deferred by `iter_with_large_drop` so dropping the tree is not
/// counted as part of building it.
fn bench_copy(c: &mut Criterion) {
    let mut g = c.benchmark_group("copy");
    for width in [1usize, 100, 10_000] {
        g.throughput(Throughput::Elements(width as u64));
        g.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, &w| {
            b.iter_with_large_drop(|| {
                let root = adder();
                let sons: Vec<_> = (0..w).map(|_| root.copy()).collect();
                (root, sons)
            });
        });
    }
    g.finish();
}

/// set(): one uncontended lock + one atomic store.
fn bench_set(c: &mut Criterion) {
    let root = adder();
    let son = root.copy();
    c.bench_function("set", |b| b.iter(|| son.set(black_box(7))));
}

/// get() over a wide tree with every son dirty: the real merge path.
/// `iter_custom` starts the clock after the sons have been re-dirtied and stops
/// it before anything is dropped, so only the merge itself is on the clock.
/// One `Instant` pair per sample is included; see the `timer_overhead` bench.
fn bench_merge_dirty(c: &mut Criterion) {
    let mut g = c.benchmark_group("merge_dirty");
    for width in [1usize, 10, 100, 1_000] {
        g.throughput(Throughput::Elements(width as u64));
        g.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, &w| {
            let root = adder();
            let sons: Vec<_> = (0..w).map(|_| root.copy()).collect();
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for s in &sons {
                        s.set(1); // off the clock
                    }
                    let t0 = Instant::now();
                    black_box(root.get());
                    total += t0.elapsed();
                }
                total
            });
        });
    }
    g.finish();
}

/// What one `Instant::now()` pair costs, so the merge_dirty figures can be read.
fn bench_timer_overhead(c: &mut Criterion) {
    c.bench_function("timer_overhead", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t0 = Instant::now();
                black_box(0u8);
                total += t0.elapsed();
            }
            total
        });
    });
}

/// get() over a tree with nothing dirty: pure polling overhead.
/// This is what a caller pays for asking when there is no work to do.
fn bench_merge_clean(c: &mut Criterion) {
    let mut g = c.benchmark_group("merge_clean");
    for width in [1usize, 10, 100, 1_000] {
        g.throughput(Throughput::Elements(width as u64));
        let root = wide(width, false);
        g.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, _| {
            b.iter(|| black_box(root.get()));
        });
    }
    g.finish();
}

/// merge recursion cost down a chain, deepest node dirty.
fn bench_merge_depth(c: &mut Criterion) {
    let mut g = c.benchmark_group("merge_depth");
    for depth in [1usize, 10, 100] {
        g.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &d| {
            let (root, leaf) = deep(d);
            b.iter(|| {
                leaf.set(1);
                black_box(root.get())
            });
        });
    }
    g.finish();
}

/// check_children(): read-only sweep, allocates a Vec per level.
fn bench_check_children(c: &mut Criterion) {
    let mut g = c.benchmark_group("check_children");
    for width in [1usize, 100, 1_000] {
        let root = wide(width, false);
        g.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, _| {
            b.iter(|| black_box(root.check_children()));
        });
    }
    g.finish();
}

/// reap: a merge that has to unlink `width` spent sons on its way out.
/// Scaled against merge_dirty at the same width, that is the cost of a pool
/// churning through short-lived workers.
fn bench_reap(c: &mut Criterion) {
    let mut g = c.benchmark_group("reap");
    for width in [1usize, 10, 100, 1_000] {
        g.throughput(Throughput::Elements(width as u64));
        g.bench_with_input(BenchmarkId::from_parameter(width), &width, |b, &w| {
            let root = adder();
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // off the clock: spawn w workers, write, drop every handle
                    for _ in 0..w {
                        root.copy().set(1);
                    }
                    let t0 = Instant::now();
                    black_box(root.get()); // folds them, then reaps them
                    total += t0.elapsed();
                }
                total
            });
        });
    }
    g.finish();
}

/// N threads each doing OPS sets on their own son while the main thread merges.
/// OPS is large enough that thread spawn is amortised away.
fn bench_contention(c: &mut Criterion) {
    const OPS: i64 = 20_000;
    let mut g = c.benchmark_group("contention");
    g.sample_size(20);
    for threads in [1usize, 2, 4, 8] {
        g.throughput(Throughput::Elements(OPS as u64 * threads as u64));
        g.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, &n| {
            b.iter(|| {
                let root = adder();
                let workers: Vec<_> = (0..n)
                    .map(|_| {
                        let son = root.copy();
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                son.set(i);
                            }
                        })
                    })
                    .collect();
                for w in workers {
                    w.join().unwrap();
                }
                black_box(root.get())
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_copy,
    bench_set,
    bench_merge_dirty,
    bench_timer_overhead,
    bench_merge_clean,
    bench_merge_depth,
    bench_check_children,
    bench_reap,
    bench_contention,
);
criterion_main!(benches);

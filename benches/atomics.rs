//! What the primitives on the write path actually cost, uncontended and with
//! every core hammering the same target. This is the measurement that explains
//! why `Weak::upgrade` had to come off the hot path.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

const OPS: usize = 200_000;

/// One thread, no one else touching the target.
fn bench_uncontended(c: &mut Criterion) {
    let mut g = c.benchmark_group("primitive_uncontended");

    let strong = Arc::new(7u64);
    let weak: Weak<u64> = Arc::downgrade(&strong);
    g.bench_function("Weak::upgrade", |b| {
        b.iter(|| black_box(weak.upgrade().is_some()))
    });

    g.bench_function("Arc::clone", |b| {
        b.iter(|| black_box(Arc::clone(&strong)))
    });

    let flag = AtomicBool::new(false);
    g.bench_function("AtomicBool::load", |b| {
        b.iter(|| black_box(flag.load(Ordering::Acquire)))
    });
    g.bench_function("AtomicBool::swap", |b| {
        b.iter(|| black_box(flag.swap(true, Ordering::AcqRel)))
    });

    let m = Mutex::new(0u64);
    g.bench_function("Mutex lock+unlock", |b| {
        b.iter(|| black_box(*m.lock().unwrap()))
    });
    g.finish();
}

/// N threads hitting the SAME target at once: the cost that matters in a tree
/// where every writer touches its ancestors.
fn bench_contended(c: &mut Criterion) {
    let mut g = c.benchmark_group("primitive_contended");
    g.sample_size(20);

    for threads in [1usize, 2, 4, 8] {
        g.throughput(Throughput::Elements((OPS * threads) as u64));

        g.bench_with_input(
            BenchmarkId::new("Weak::upgrade", threads),
            &threads,
            |b, &n| {
                let strong = Arc::new(7u64);
                b.iter_custom(|iters| run(n, iters, || Arc::downgrade(&strong), |w| {
                    black_box(w.upgrade().is_some());
                }))
            },
        );

        g.bench_with_input(
            BenchmarkId::new("AtomicBool::load", threads),
            &threads,
            |b, &n| {
                let flag = Arc::new(AtomicBool::new(true));
                b.iter_custom(|iters| run(n, iters, || Arc::clone(&flag), |f| {
                    black_box(f.load(Ordering::Acquire));
                }))
            },
        );

        g.bench_with_input(
            BenchmarkId::new("AtomicUsize::fetch_add", threads),
            &threads,
            |b, &n| {
                let ctr = Arc::new(AtomicUsize::new(0));
                b.iter_custom(|iters| run(n, iters, || Arc::clone(&ctr), |c| {
                    black_box(c.fetch_add(1, Ordering::Relaxed));
                }))
            },
        );
    }
    g.finish();
}

/// Spawns `threads` workers behind a barrier and times only the hammering.
fn run<T: Send + 'static>(
    threads: usize,
    iters: u64,
    make: impl Fn() -> T,
    body: impl Fn(&T) + Copy + Send + 'static,
) -> Duration {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let t = make();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..(iters as usize * OPS / 1000) {
                    body(&t);
                }
                barrier.wait();
            })
        })
        .collect();
    barrier.wait();
    let t0 = Instant::now();
    barrier.wait();
    let dt = t0.elapsed();
    for h in hs {
        h.join().unwrap();
    }
    dt
}

criterion_group!(atomics, bench_uncontended, bench_contended);
criterion_main!(atomics);

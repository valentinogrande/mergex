//! mergex against the obvious std alternatives, same workload for all.
//!
//! Every entrant performs the same operation -- a read-modify-write of an i64,
//! `+= i`, twenty thousand times per thread. What differs is the topology: mergex,
//! `sharded` and `plain_local` hit a slot the thread owns, while `Mutex`,
//! `RwLock` and `AtomicI64` hit one cell shared by every thread, which is the
//! only way a single cell can hold a running total. That difference is the
//! subject of the benchmark.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use left_right::{Absorb, ReadHandle, WriteHandle};
use mergex::Mergex;
use std::hint::black_box;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

const OPS: i64 = 20_000;

/// A shard on its own cache line, to separate contention from false sharing.
#[repr(align(128))]
struct Padded(Mutex<i64>);

/// left-right's model: one writer, many wait-free readers, two copies of the
/// value swapped on publish. Writers must serialise through the single
/// WriteHandle, so N of them need a Mutex around it.
#[derive(Clone)]
struct Counter(i64);
enum Add {
    By(i64),
}
impl Absorb<Add> for Counter {
    fn absorb_first(&mut self, op: &mut Add, _: &Self) {
        let Add::By(v) = *op;
        self.0 += v;
    }
    fn absorb_second(&mut self, op: Add, _: &Self) {
        let Add::By(v) = op;
        self.0 += v;
    }
    fn drop_first(self: Box<Self>) {}
    fn sync_with(&mut self, first: &Self) {
        self.0 = first.0;
    }
}
fn left_right_pair() -> (WriteHandle<Counter, Add>, ReadHandle<Counter>) {
    left_right::new_from_empty::<Counter, Add>(Counter(0))
}

/// N threads, OPS writes each, then read the aggregate once.
fn bench_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("write_throughput");
    g.sample_size(20);

    for threads in [1usize, 2, 4, 8] {
        g.throughput(Throughput::Elements(OPS as u64 * threads as u64));

        g.bench_with_input(BenchmarkId::new("mergex", threads), &threads, |b, &n| {
            b.iter(|| {
                let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let son = root.copy();
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                son.update(|slot| *slot += i);
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(root.get())
            });
        });

        g.bench_with_input(BenchmarkId::new("Mutex", threads), &threads, |b, &n| {
            b.iter(|| {
                let m = Arc::new(Mutex::new(0i64));
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                *m.lock().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(*m.lock().unwrap())
            });
        });

        g.bench_with_input(BenchmarkId::new("RwLock", threads), &threads, |b, &n| {
            b.iter(|| {
                let m = Arc::new(RwLock::new(0i64));
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                *m.write().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(*m.read().unwrap())
            });
        });

        g.bench_with_input(BenchmarkId::new("Atomic", threads), &threads, |b, &n| {
            b.iter(|| {
                let a = Arc::new(AtomicI64::new(0));
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let a = Arc::clone(&a);
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                a.fetch_add(i, Ordering::Relaxed);
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(a.load(Ordering::Acquire))
            });
        });

        // one Mutex per thread, summed at the end: mergex by hand, one level deep
        g.bench_with_input(BenchmarkId::new("sharded", threads), &threads, |b, &n| {
            b.iter(|| {
                let shards: Arc<Vec<Mutex<i64>>> =
                    Arc::new((0..n).map(|_| Mutex::new(0)).collect());
                let hs: Vec<_> = (0..n)
                    .map(|k| {
                        let s = Arc::clone(&shards);
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                *s[k].lock().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(shards.iter().map(|m| *m.lock().unwrap()).sum::<i64>())
            });
        });

        // the floor: one plain i64 per thread, zero synchronisation, folded on join
        g.bench_with_input(BenchmarkId::new("plain_local", threads), &threads, |b, &n| {
            b.iter(|| {
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        thread::spawn(move || {
                            let mut local = 0i64;
                            for i in 1..=OPS {
                                local += i;
                                black_box(&local); // force the store, keep it honest
                            }
                            local
                        })
                    })
                    .collect();
                black_box(hs.into_iter().map(|h| h.join().unwrap()).sum::<i64>())
            });
        });

        // left-right: writers serialise on the one WriteHandle, publishing in
        // batches of 1000 so the copy-back is amortised the way its design intends
        g.bench_with_input(BenchmarkId::new("left_right", threads), &threads, |b, &n| {
            b.iter(|| {
                let (w, r) = left_right_pair();
                let w = Arc::new(Mutex::new(w));
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let w = Arc::clone(&w);
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                let mut g = w.lock().unwrap();
                                g.append(Add::By(i));
                                if i % 1000 == 0 {
                                    g.publish();
                                }
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                w.lock().unwrap().publish();
                black_box(r.enter().map(|g| g.0).unwrap_or(0))
            });
        });

        // sharded again, but each shard on its own cache line
        g.bench_with_input(BenchmarkId::new("sharded_padded", threads), &threads, |b, &n| {
            b.iter(|| {
                let shards: Arc<Vec<Padded>> =
                    Arc::new((0..n).map(|_| Padded(Mutex::new(0))).collect());
                let hs: Vec<_> = (0..n)
                    .map(|k| {
                        let s = Arc::clone(&shards);
                        thread::spawn(move || {
                            for i in 1..=OPS {
                                *s[k].0.lock().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(shards.iter().map(|m| *m.0.lock().unwrap()).sum::<i64>())
            });
        });
    }
    g.finish();
}

/// Uncontended single-value read latency: what a worker pays to look at its own data.
fn bench_read_local(c: &mut Criterion) {
    let mut g = c.benchmark_group("read_local");

    let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
    let son = root.copy();
    son.set(7); // so it holds what every other entrant below holds
    g.bench_function("mergex_get_threaded", |b| {
        b.iter(|| black_box(son.get_threaded()))
    });

    let m = Mutex::new(7i64);
    g.bench_function("Mutex", |b| b.iter(|| black_box(*m.lock().unwrap())));

    let r = RwLock::new(7i64);
    g.bench_function("RwLock", |b| b.iter(|| black_box(*r.read().unwrap())));

    let a = AtomicI64::new(7);
    g.bench_function("Atomic", |b| {
        b.iter(|| black_box(a.load(Ordering::Acquire)))
    });

    let (mut w, r) = left_right_pair();
    w.append(Add::By(7));
    w.publish();
    g.bench_function("left_right", |b| {
        b.iter(|| black_box(r.enter().map(|g| g.0).unwrap_or(0)))
    });
    g.finish();
}

/// Reading the aggregate when nothing changed since the last read.
/// Scaled by how many producers exist, because some designs must sweep them all.
fn bench_read_global_clean(c: &mut Criterion) {
    let mut g = c.benchmark_group("read_global_clean");

    for producers in [8usize, 64, 512] {
        let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
        let sons: Vec<_> = (0..producers).map(|_| root.copy()).collect();
        for s in &sons {
            s.set(1);
        }
        root.get(); // settle, so the tree is clean
        g.bench_with_input(
            BenchmarkId::new("mergex", producers),
            &producers,
            |b, _| b.iter(|| black_box(root.get())),
        );

        let m = Mutex::new(7i64);
        g.bench_with_input(BenchmarkId::new("Mutex", producers), &producers, |b, _| {
            b.iter(|| black_box(*m.lock().unwrap()))
        });

        let a = AtomicI64::new(7);
        g.bench_with_input(BenchmarkId::new("Atomic", producers), &producers, |b, _| {
            b.iter(|| black_box(a.load(Ordering::Acquire)))
        });

        let (mut w, rh) = left_right_pair();
        w.append(Add::By(7));
        w.publish();
        g.bench_with_input(
            BenchmarkId::new("left_right", producers),
            &producers,
            |b, _| b.iter(|| black_box(rh.enter().map(|g| g.0).unwrap_or(0))),
        );

        let shards: Vec<Mutex<i64>> = (0..producers).map(|_| Mutex::new(1)).collect();
        g.bench_with_input(BenchmarkId::new("sharded", producers), &producers, |b, _| {
            b.iter(|| black_box(shards.iter().map(|m| *m.lock().unwrap()).sum::<i64>()))
        });
    }
    g.finish();
}

criterion_group!(
    comparison,
    bench_write,
    bench_read_local,
    bench_read_global_clean
);
criterion_main!(comparison);

//! The read side, measured where it actually costs something.
//!
//! `comparison.rs` has three read benches and all three flatter mergex:
//! `read_local` reads a node's own slot under an uncontended lock,
//! `read_global_clean` hits the `subtree_dirty` fast path and returns in O(1),
//! and `benches/mergex.rs::merge_dirty` does walk the expensive path but has no
//! competitor next to it. So the number that hurts never appears beside
//! `left-right`. That reads as advocacy, and a sceptical reader spots it in
//! thirty seconds.
//!
//! This file is the other half. A `get()` at the root over W dirty sons costs:
//!
//!   1 swap (subtree_dirty)
//!   1 alloc + W Arc::clone            <- snapshot_sons
//!   per son: 1 swap + 1 load + lock(son) + mem::replace + lock(father) + merge
//!
//! -- so O(W) locks, and note the father's lock is taken and released once per
//! son, inside the loop. `left-right` reads with one atomic load and no lock at
//! all. The gap is orders of magnitude, and it belongs in the same table.
//!
//! Four things are measured here:
//!
//!   read_global_dirty   the honest twin of read_global_clean
//!   rw_ratio            the crossover: how many writes per read it takes
//!                       before mergex is the right answer
//!   write_under_reader  what a live reader costs the writers it is supposed
//!                       not to disturb
//!   big_value           the same story with T = HashMap, not i64

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use left_right::{Absorb, ReadHandle, WriteHandle};
use mergex::Mergex;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// A shard on its own cache line, to separate contention from false sharing.
#[repr(align(128))]
struct Padded(Mutex<i64>);

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

fn adder() -> Mergex<i64> {
    Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s)
}

// ---------------------------------------------------------------- (a) dirty read

/// Reading the aggregate when every producer has written since the last read.
///
/// The twin of `read_global_clean`, and the one that should be quoted next to
/// it. Producers are re-dirtied off the clock, as in `merge_dirty`, so only the
/// read is timed. Scaled by producer count because that is what mergex and
/// `sharded` have to sweep and the single-cell designs do not.
fn bench_read_global_dirty(c: &mut Criterion) {
    let mut g = c.benchmark_group("read_global_dirty");

    for producers in [8usize, 64, 512] {
        g.throughput(Throughput::Elements(producers as u64));

        let root = adder();
        let sons: Vec<_> = (0..producers).map(|_| root.copy()).collect();
        g.bench_with_input(BenchmarkId::new("mergex", producers), &producers, |b, _| {
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

        let shards: Vec<Padded> = (0..producers).map(|_| Padded(Mutex::new(0))).collect();
        g.bench_with_input(BenchmarkId::new("sharded", producers), &producers, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for s in &shards {
                        *s.0.lock().unwrap() += 1;
                    }
                    let t0 = Instant::now();
                    black_box(shards.iter().map(|s| *s.0.lock().unwrap()).sum::<i64>());
                    total += t0.elapsed();
                }
                total
            });
        });

        // The single-cell designs have nothing to sweep: the producers paid on
        // the write side instead. They still do the same W writes off the clock
        // so the workload matches, and their read stays O(1). That contrast is
        // the whole point of the bench.
        let m = Mutex::new(0i64);
        g.bench_with_input(BenchmarkId::new("Mutex", producers), &producers, |b, &n| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..n {
                        *m.lock().unwrap() += 1;
                    }
                    let t0 = Instant::now();
                    black_box(*m.lock().unwrap());
                    total += t0.elapsed();
                }
                total
            });
        });

        let a = AtomicI64::new(0);
        g.bench_with_input(BenchmarkId::new("Atomic", producers), &producers, |b, &n| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for _ in 0..n {
                        a.fetch_add(1, Ordering::Relaxed);
                    }
                    let t0 = Instant::now();
                    black_box(a.load(Ordering::Acquire));
                    total += t0.elapsed();
                }
                total
            });
        });

        let (mut w, rh) = left_right_pair();
        g.bench_with_input(
            BenchmarkId::new("left_right", producers),
            &producers,
            |b, &n| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        for _ in 0..n {
                            w.append(Add::By(1));
                        }
                        w.publish(); // publishing is write-side work, off the clock
                        let t0 = Instant::now();
                        black_box(rh.enter().map(|g| g.0).unwrap_or(0));
                        total += t0.elapsed();
                    }
                    total
                });
            },
        );
    }
    g.finish();
}

// ---------------------------------------------------------------- (b) ratio sweep

const RATIO_OPS: i64 = 5_000;
const RATIO_WRITERS: usize = 4;

/// The decisive chart: writes per read, swept.
///
/// Four writer threads run `RATIO_OPS` writes each while one reader thread does
/// exactly `total_writes / ratio` reads of the aggregate. Throughput counts
/// both, so a single curve per entrant says where each design belongs. There is
/// a crossover ratio at which mergex stops being the right answer; publishing
/// that number is the honest positioning of the crate, and without it
/// "use it for write-heavy workloads" is an opinion rather than a measurement.
///
/// Thread spawn sits inside the timed region, as it does in
/// `comparison.rs::bench_write`. It is a constant across entrants, so it raises
/// the noise floor without tilting the comparison.
fn bench_rw_ratio(c: &mut Criterion) {
    let mut g = c.benchmark_group("rw_ratio");
    g.sample_size(10);

    let writes = RATIO_OPS as usize * RATIO_WRITERS;

    for ratio in [10_000usize, 100, 1] {
        let reads = (writes / ratio).max(1);
        g.throughput(Throughput::Elements((writes + reads) as u64));

        g.bench_with_input(BenchmarkId::new("mergex", ratio), &reads, |b, &reads| {
            b.iter(|| {
                let root = adder();
                let ws: Vec<_> = (0..RATIO_WRITERS)
                    .map(|_| {
                        let son = root.copy();
                        thread::spawn(move || {
                            for i in 1..=RATIO_OPS {
                                son.update(|v| *v += i);
                            }
                        })
                    })
                    .collect();
                let r = root.clone();
                let rd = thread::spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..reads {
                        acc = acc.wrapping_add(r.get());
                    }
                    acc
                });
                for w in ws {
                    w.join().unwrap();
                }
                black_box(rd.join().unwrap());
                black_box(root.get())
            });
        });

        g.bench_with_input(BenchmarkId::new("sharded", ratio), &reads, |b, &reads| {
            b.iter(|| {
                let shards: Arc<Vec<Padded>> =
                    Arc::new((0..RATIO_WRITERS).map(|_| Padded(Mutex::new(0))).collect());
                let ws: Vec<_> = (0..RATIO_WRITERS)
                    .map(|k| {
                        let s = Arc::clone(&shards);
                        thread::spawn(move || {
                            for i in 1..=RATIO_OPS {
                                *s[k].0.lock().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                let s = Arc::clone(&shards);
                let rd = thread::spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..reads {
                        acc = acc.wrapping_add(s.iter().map(|m| *m.0.lock().unwrap()).sum::<i64>());
                    }
                    acc
                });
                for w in ws {
                    w.join().unwrap();
                }
                black_box(rd.join().unwrap());
                black_box(shards.iter().map(|m| *m.0.lock().unwrap()).sum::<i64>())
            });
        });

        g.bench_with_input(BenchmarkId::new("Mutex", ratio), &reads, |b, &reads| {
            b.iter(|| {
                let m = Arc::new(Mutex::new(0i64));
                let ws: Vec<_> = (0..RATIO_WRITERS)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for i in 1..=RATIO_OPS {
                                *m.lock().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                let r = Arc::clone(&m);
                let rd = thread::spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..reads {
                        acc = acc.wrapping_add(*r.lock().unwrap());
                    }
                    acc
                });
                for w in ws {
                    w.join().unwrap();
                }
                black_box(rd.join().unwrap());
                black_box(*m.lock().unwrap())
            });
        });

        g.bench_with_input(BenchmarkId::new("RwLock", ratio), &reads, |b, &reads| {
            b.iter(|| {
                let m = Arc::new(RwLock::new(0i64));
                let ws: Vec<_> = (0..RATIO_WRITERS)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for i in 1..=RATIO_OPS {
                                *m.write().unwrap() += i;
                            }
                        })
                    })
                    .collect();
                let r = Arc::clone(&m);
                let rd = thread::spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..reads {
                        acc = acc.wrapping_add(*r.read().unwrap());
                    }
                    acc
                });
                for w in ws {
                    w.join().unwrap();
                }
                black_box(rd.join().unwrap());
                black_box(*m.read().unwrap())
            });
        });

        g.bench_with_input(BenchmarkId::new("Atomic", ratio), &reads, |b, &reads| {
            b.iter(|| {
                let a = Arc::new(AtomicI64::new(0));
                let ws: Vec<_> = (0..RATIO_WRITERS)
                    .map(|_| {
                        let a = Arc::clone(&a);
                        thread::spawn(move || {
                            for i in 1..=RATIO_OPS {
                                a.fetch_add(i, Ordering::Relaxed);
                            }
                        })
                    })
                    .collect();
                let r = Arc::clone(&a);
                let rd = thread::spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..reads {
                        acc = acc.wrapping_add(r.load(Ordering::Acquire));
                    }
                    acc
                });
                for w in ws {
                    w.join().unwrap();
                }
                black_box(rd.join().unwrap());
                black_box(a.load(Ordering::Acquire))
            });
        });

        // left-right is the design this ratio sweep exists to be measured
        // against: its readers are wait-free, so it should win by more and more
        // as the ratio drops towards 1.
        g.bench_with_input(BenchmarkId::new("left_right", ratio), &reads, |b, &reads| {
            b.iter(|| {
                let (w, rh) = left_right_pair();
                let w = Arc::new(Mutex::new(w));
                let ws: Vec<_> = (0..RATIO_WRITERS)
                    .map(|_| {
                        let w = Arc::clone(&w);
                        thread::spawn(move || {
                            for i in 1..=RATIO_OPS {
                                let mut g = w.lock().unwrap();
                                g.append(Add::By(i));
                                if i % 1000 == 0 {
                                    g.publish();
                                }
                            }
                        })
                    })
                    .collect();
                let rd = thread::spawn(move || {
                    let mut acc = 0i64;
                    for _ in 0..reads {
                        acc = acc.wrapping_add(rh.enter().map(|g| g.0).unwrap_or(0));
                    }
                    acc
                });
                for h in ws {
                    h.join().unwrap();
                }
                w.lock().unwrap().publish();
                black_box(rd.join().unwrap())
            });
        });
    }
    g.finish();
}

// ------------------------------------------------- (c) reader/writer interference

const INTERFERE_OPS: i64 = 20_000;

/// Runs `writer` on `threads` threads and times only their run.
///
/// A barrier holds the workers on the start line and a second one releases when
/// the last is done, so spawn and join are off the clock. When `reader` is
/// given it spins for the whole timed region and is stopped afterwards; its own
/// time is never counted, only the drag it puts on the writers.
fn timed_writers(
    threads: usize,
    writer: Arc<dyn Fn(usize) + Send + Sync>,
    reader: Option<Box<dyn FnOnce(Arc<AtomicBool>) + Send>>,
) -> Duration {
    let stop = Arc::new(AtomicBool::new(false));
    let gate = Arc::new(Barrier::new(threads + 1));
    let done = Arc::new(Barrier::new(threads + 1));

    let rh = reader.map(|f| {
        let stop = Arc::clone(&stop);
        thread::spawn(move || f(stop))
    });

    let hs: Vec<_> = (0..threads)
        .map(|k| {
            let (w, gate, done) = (Arc::clone(&writer), Arc::clone(&gate), Arc::clone(&done));
            thread::spawn(move || {
                gate.wait();
                w(k);
                done.wait();
            })
        })
        .collect();

    gate.wait();
    let t0 = Instant::now();
    done.wait();
    let dt = t0.elapsed();

    stop.store(true, Ordering::Relaxed);
    for h in hs {
        h.join().unwrap();
    }
    if let Some(h) = rh {
        h.join().unwrap();
    }
    dt
}

/// Writer throughput with and without a reader running alongside.
///
/// mergex's pitch is that writers never share a cache line. That holds only
/// while nobody is reading: `merge_sons` takes each son's `data` lock, which is
/// the very lock `update()` needs, so a hot reader serialises against every
/// writer it walks past. This is the bench that puts a number on it, and the
/// entrants it should be read against are `sharded` -- which has the same
/// problem -- and `left-right`, whose readers are wait-free and therefore
/// should show no gap at all.
fn bench_write_under_reader(c: &mut Criterion) {
    let mut g = c.benchmark_group("write_under_reader");
    g.sample_size(20);

    for threads in [4usize, 8] {
        g.throughput(Throughput::Elements(INTERFERE_OPS as u64 * threads as u64));

        for &(tag, with_reader) in &[("quiet", false), ("reader", true)] {
            let id = format!("{threads}t_{tag}");

            g.bench_function(BenchmarkId::new("mergex", &id), |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let root = adder();
                        let sons: Arc<Vec<Mergex<i64>>> =
                            Arc::new((0..threads).map(|_| root.copy()).collect());
                        let w = {
                            let sons = Arc::clone(&sons);
                            Arc::new(move |k: usize| {
                                for i in 1..=INTERFERE_OPS {
                                    sons[k].update(|v| *v += i);
                                }
                            })
                        };
                        let r: Option<Box<dyn FnOnce(Arc<AtomicBool>) + Send>> = if with_reader {
                            let root = root.clone();
                            Some(Box::new(move |stop: Arc<AtomicBool>| {
                                while !stop.load(Ordering::Relaxed) {
                                    black_box(root.get());
                                }
                            }))
                        } else {
                            None
                        };
                        total += timed_writers(threads, w, r);
                        black_box(root.get());
                    }
                    total
                });
            });

            g.bench_function(BenchmarkId::new("sharded", &id), |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let shards: Arc<Vec<Padded>> =
                            Arc::new((0..threads).map(|_| Padded(Mutex::new(0))).collect());
                        let w = {
                            let s = Arc::clone(&shards);
                            Arc::new(move |k: usize| {
                                for i in 1..=INTERFERE_OPS {
                                    *s[k].0.lock().unwrap() += i;
                                }
                            })
                        };
                        let r: Option<Box<dyn FnOnce(Arc<AtomicBool>) + Send>> = if with_reader {
                            let s = Arc::clone(&shards);
                            Some(Box::new(move |stop: Arc<AtomicBool>| {
                                while !stop.load(Ordering::Relaxed) {
                                    black_box(s.iter().map(|m| *m.0.lock().unwrap()).sum::<i64>());
                                }
                            }))
                        } else {
                            None
                        };
                        total += timed_writers(threads, w, r);
                    }
                    total
                });
            });

            g.bench_function(BenchmarkId::new("left_right", &id), |b| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let (wh, rh) = left_right_pair();
                        let wh = Arc::new(Mutex::new(wh));
                        let w = {
                            let wh = Arc::clone(&wh);
                            Arc::new(move |_k: usize| {
                                for i in 1..=INTERFERE_OPS {
                                    let mut g = wh.lock().unwrap();
                                    g.append(Add::By(i));
                                    if i % 1000 == 0 {
                                        g.publish();
                                    }
                                }
                            })
                        };
                        let r: Option<Box<dyn FnOnce(Arc<AtomicBool>) + Send>> = if with_reader {
                            Some(Box::new(move |stop: Arc<AtomicBool>| {
                                while !stop.load(Ordering::Relaxed) {
                                    black_box(rh.enter().map(|g| g.0).unwrap_or(0));
                                }
                            }))
                        } else {
                            None
                        };
                        total += timed_writers(threads, w, r);
                    }
                    total
                });
            });
        }
    }
    g.finish();
}

// ---------------------------------------------------------------- (d) big value

const KEYS: u64 = 256;
const MAP_OPS: u64 = 20_000;
type Map = HashMap<u64, u64>;

fn map_root() -> Mergex<Map> {
    Mergex::new(Map::new(), Map::new(), |f: &mut Map, s: &Map| {
        for (k, v) in s {
            *f.entry(*k).or_insert(0) += v;
        }
    })
}

/// The same story with a value that is not eight bytes.
///
/// Everything else in the suite measures `i64`, where the merge closure is one
/// instruction and the identity is free. With `T = HashMap` the shape changes:
/// the fold walks the son's map key by key, `get()` clones the whole aggregate
/// out through `get_threaded`, and each node holds a full map of its own -- so
/// the memory is O(nodes x |T|), not O(nodes). That is the realistic shape of
/// the use case mergex is actually for (per-thread aggregation maps), so it is
/// the shape the numbers should be quoted in.
///
/// No `left-right` entrant here: it would need its own `Absorb` implementation
/// over the map, which measures that implementation as much as the design.
fn bench_big_value(c: &mut Criterion) {
    let mut g = c.benchmark_group("big_value_write");
    g.sample_size(20);

    for threads in [1usize, 4, 8] {
        g.throughput(Throughput::Elements(MAP_OPS * threads as u64));

        g.bench_with_input(BenchmarkId::new("mergex", threads), &threads, |b, &n| {
            b.iter(|| {
                let root = map_root();
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let son = root.copy();
                        thread::spawn(move || {
                            for i in 0..MAP_OPS {
                                son.update(|m| *m.entry(i % KEYS).or_insert(0) += 1);
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(root.get().len())
            });
        });

        g.bench_with_input(BenchmarkId::new("Mutex", threads), &threads, |b, &n| {
            b.iter(|| {
                let m = Arc::new(Mutex::new(Map::new()));
                let hs: Vec<_> = (0..n)
                    .map(|_| {
                        let m = Arc::clone(&m);
                        thread::spawn(move || {
                            for i in 0..MAP_OPS {
                                *m.lock().unwrap().entry(i % KEYS).or_insert(0) += 1;
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                black_box(m.lock().unwrap().len())
            });
        });

        g.bench_with_input(BenchmarkId::new("sharded", threads), &threads, |b, &n| {
            b.iter(|| {
                let shards: Arc<Vec<Mutex<Map>>> =
                    Arc::new((0..n).map(|_| Mutex::new(Map::new())).collect());
                let hs: Vec<_> = (0..n)
                    .map(|k| {
                        let s = Arc::clone(&shards);
                        thread::spawn(move || {
                            for i in 0..MAP_OPS {
                                *s[k].lock().unwrap().entry(i % KEYS).or_insert(0) += 1;
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
                let mut agg = Map::new();
                for s in shards.iter() {
                    for (k, v) in s.lock().unwrap().iter() {
                        *agg.entry(*k).or_insert(0) += v;
                    }
                }
                black_box(agg.len())
            });
        });
    }
    g.finish();

    // The read half: the aggregate rebuilt over W producers, each holding a
    // full map. This is where the per-fold identity clone and the final
    // `get_threaded` clone of the whole aggregate both land.
    let mut g = c.benchmark_group("big_value_read");
    for producers in [8usize, 64] {
        g.throughput(Throughput::Elements(producers as u64));

        let root = map_root();
        let sons: Vec<_> = (0..producers).map(|_| root.copy()).collect();
        for s in &sons {
            s.update(|m| {
                for k in 0..KEYS {
                    m.insert(k, 1);
                }
            });
        }
        g.bench_with_input(BenchmarkId::new("mergex", producers), &producers, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for s in &sons {
                        s.update(|m| {
                            for k in 0..KEYS {
                                *m.entry(k).or_insert(0) += 1;
                            }
                        });
                    }
                    let t0 = Instant::now();
                    black_box(root.get().len());
                    total += t0.elapsed();
                }
                total
            });
        });

        let shards: Vec<Mutex<Map>> = (0..producers)
            .map(|_| Mutex::new((0..KEYS).map(|k| (k, 1u64)).collect()))
            .collect();
        g.bench_with_input(BenchmarkId::new("sharded", producers), &producers, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    for s in &shards {
                        let mut m = s.lock().unwrap();
                        for k in 0..KEYS {
                            *m.entry(k).or_insert(0) += 1;
                        }
                    }
                    let t0 = Instant::now();
                    let mut agg = Map::new();
                    for s in &shards {
                        for (k, v) in s.lock().unwrap().iter() {
                            *agg.entry(*k).or_insert(0) += v;
                        }
                    }
                    black_box(agg.len());
                    total += t0.elapsed();
                }
                total
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_read_global_dirty,
    bench_rw_ratio,
    bench_write_under_reader,
    bench_big_value,
);
criterion_main!(benches);

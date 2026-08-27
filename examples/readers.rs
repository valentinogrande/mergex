//! Concurrent read throughput -- the ground left-right is actually built for.
//!
//! A single-reader latency number understates a wait-free reader badly: the
//! whole point of left-right is that N readers never queue behind each other or
//! behind the writer. So this measures N reader threads hammering the aggregate,
//! twice: with the tree quiet, and with one writer running the whole time.

use left_right::{Absorb, ReadHandle, WriteHandle};
use mergex::Mergex;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::Instant;

/// Builds the optional background writer thread for a measurement.
type WriterFactory = Box<dyn FnOnce(Arc<AtomicBool>) -> thread::JoinHandle<()>>;

const READS: usize = 200_000;
const RUNS: usize = 9;
const THREADS: [usize; 4] = [1, 2, 4, 8];

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

/// Runs `n` reader threads through `read`, timing only the reads. When `writer`
/// is Some, it runs alongside for the whole measurement and is stopped after.
fn measure<T: Clone + Send + 'static>(
    n: usize,
    target: T,
    read: fn(&T) -> i64,
    writer: Option<WriterFactory>,
) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let w = writer.map(|f| f(Arc::clone(&stop)));

    let barrier = Arc::new(Barrier::new(n + 1));
    let hs: Vec<_> = (0..n)
        .map(|_| {
            let t = target.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..READS {
                    black_box(read(&t));
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
    stop.store(true, Ordering::Relaxed);
    if let Some(w) = w {
        w.join().unwrap();
    }
    (READS as f64 * n as f64) / dt.as_secs_f64() / 1e6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    for with_writer in [false, true] {
        println!(
            "\n=== concurrent readers, {} ===  (M reads/s, median of {} runs)",
            if with_writer { "one writer running alongside" } else { "tree quiet" },
            RUNS
        );
        println!("{:<16}{:>10}{:>10}{:>10}{:>10}", "structure", "1", "2", "4", "8");

        for name in ["Atomic", "Mutex", "RwLock", "mergex", "left_right"] {
            let mut row = format!("{:<16}", name);
            for n in THREADS {
                let mut rates = Vec::with_capacity(RUNS);
                for _ in 0..RUNS {
                    rates.push(match name {
                        "Atomic" => {
                            let a = Arc::new(AtomicI64::new(7));
                            let wa = Arc::clone(&a);
                            measure(n, a, |a| a.load(Ordering::Acquire), wr(with_writer, move |s| {
                                while !s.load(Ordering::Relaxed) {
                                    wa.fetch_add(1, Ordering::Relaxed);
                                }
                            }))
                        }
                        "Mutex" => {
                            let m = Arc::new(Mutex::new(7i64));
                            let wm = Arc::clone(&m);
                            measure(n, m, |m| *m.lock().unwrap(), wr(with_writer, move |s| {
                                while !s.load(Ordering::Relaxed) {
                                    *wm.lock().unwrap() += 1;
                                }
                            }))
                        }
                        "RwLock" => {
                            let m = Arc::new(RwLock::new(7i64));
                            let wm = Arc::clone(&m);
                            measure(n, m, |m| *m.read().unwrap(), wr(with_writer, move |s| {
                                while !s.load(Ordering::Relaxed) {
                                    *wm.write().unwrap() += 1;
                                }
                            }))
                        }
                        "mergex" => {
                            let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
                            let son = root.copy();
                            root.get();
                            measure(n, root, |r| r.get(), wr(with_writer, move |s| {
                                while !s.load(Ordering::Relaxed) {
                                    son.update(|v| *v += 1);
                                }
                            }))
                        }
                        "left_right" => {
                            let (mut w, r) =
                                left_right::new_from_empty::<Counter, Add>(Counter(0));
                            w.append(Add::By(7));
                            w.publish();
                            let rf = r.clone();
                            measure(
                                n,
                                rf,
                                |r: &ReadHandle<Counter>| r.enter().map(|g| g.0).unwrap_or(0),
                                wr(with_writer, move |s| {
                                    let mut w: WriteHandle<Counter, Add> = w;
                                    while !s.load(Ordering::Relaxed) {
                                        w.append(Add::By(1));
                                        w.publish();
                                    }
                                }),
                            )
                        }
                        _ => unreachable!(),
                    });
                }
                row.push_str(&format!("{:>10.1}", median(rates)));
            }
            println!("{row}");
        }
    }
}

/// Wraps a writer loop into the optional-thread shape `measure` wants.
fn wr(
    enabled: bool,
    body: impl FnOnce(Arc<AtomicBool>) + Send + 'static,
) -> Option<WriterFactory> {
    if !enabled {
        return None;
    }
    Some(Box::new(move |stop| thread::spawn(move || body(stop))))
}

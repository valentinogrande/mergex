//! The same comparison as `repeat.rs`, but with a `T` that is not free to fold.
//!
//! Every benchmark in this crate used to run on `i64`, where merging is one
//! instruction -- the case most favourable to a plain shard array, because
//! mergex's per-node machinery is then pure overhead. This one uses a
//! `HashMap<u64, u64>`, where merging costs something and a shared lock has to
//! hold it for the whole update.
//!
//! `thread_local::ThreadLocal<T>` is included because it is the closest thing
//! in the Rust ecosystem: per-thread slots with dynamic membership, aggregated
//! by iteration. Two variants, because the difference is the whole story of
//! what it costs to find your slot -- `thread_local` looks the slot up on every
//! operation, which is how it is normally written; `thread_local_hoisted` looks
//! it up once outside the loop, which is its floor.

use mergex::Mergex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use thread_local::ThreadLocal;

type Map = HashMap<u64, u64>;

/// A thread_local slot on its own cache line. `ThreadLocal` packs its slots
/// contiguously, so neighbouring threads share a line and fight over it; on
/// `i64` that costs 15x past four threads. Padding is the fair comparison.
#[repr(align(128))]
#[derive(Default)]
struct PaddedCell(RefCell<Map>);

const OPS: u64 = 50_000;
const KEYS: u64 = 64;
const RUNS: usize = 9;
const THREADS: [usize; 4] = [1, 2, 4, 8];

fn fold(father: &mut Map, son: &Map) {
    for (k, v) in son {
        *father.entry(*k).or_insert(0) += v;
    }
}

/// Holds every worker on the start line, times only the work.
fn timed(n: usize, body: impl Fn(usize) + Send + Sync + 'static + Clone) -> Duration {
    let barrier = Arc::new(Barrier::new(n + 1));
    let hs: Vec<_> = (0..n)
        .map(|k| {
            let barrier = Arc::clone(&barrier);
            let body = body.clone();
            thread::spawn(move || {
                barrier.wait();
                body(k);
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!(
        "T = HashMap<u64, u64>, {} keys, {} updates per thread, median of {} runs",
        KEYS, OPS, RUNS
    );
    println!("million updates per second, higher is better\n");
    println!("{:<22}{:>10}{:>10}{:>10}{:>10}", "structure", "1", "2", "4", "8");

    for name in [
        "mergex",
        "thread_local_hoisted",
        "thread_local",
        "thread_local_padded",
        "sharded",
        "Mutex",
    ] {
        let mut row = format!("{:<22}", name);
        for n in THREADS {
            let mut rates = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let dt = match name {
                    "mergex" => {
                        let root = Mergex::new(Map::new(), Map::new(), fold);
                        let sons: Vec<_> = (0..n).map(|_| root.copy()).collect();
                        let sons = Arc::new(sons);
                        let d = timed(n, move |k| {
                            for i in 0..OPS {
                                sons[k].update(|m| *m.entry(i % KEYS).or_insert(0) += 1);
                            }
                        });
                        black_box(root.get().len());
                        d
                    }
                    "Mutex" => {
                        let m = Arc::new(Mutex::new(Map::new()));
                        let m2 = Arc::clone(&m);
                        let d = timed(n, move |_| {
                            for i in 0..OPS {
                                *m2.lock().unwrap().entry(i % KEYS).or_insert(0) += 1;
                            }
                        });
                        black_box(m.lock().unwrap().len());
                        d
                    }
                    "sharded" => {
                        let s: Arc<Vec<Mutex<Map>>> =
                            Arc::new((0..n).map(|_| Mutex::new(Map::new())).collect());
                        let s2 = Arc::clone(&s);
                        let d = timed(n, move |k| {
                            for i in 0..OPS {
                                *s2[k].lock().unwrap().entry(i % KEYS).or_insert(0) += 1;
                            }
                        });
                        let mut agg = Map::new();
                        for sh in s.iter() {
                            fold(&mut agg, &sh.lock().unwrap());
                        }
                        black_box(agg.len());
                        d
                    }
                    // the slot is found again on every single update
                    "thread_local" => {
                        let tls: Arc<ThreadLocal<RefCell<Map>>> = Arc::new(ThreadLocal::new());
                        let t2 = Arc::clone(&tls);
                        let d = timed(n, move |_| {
                            for i in 0..OPS {
                                *t2.get_or_default()
                                    .borrow_mut()
                                    .entry(i % KEYS)
                                    .or_insert(0) += 1;
                            }
                        });
                        // the workers are joined, so this is the only handle left;
                        // consuming it avoids needing T: Sync just to aggregate
                        let mut agg = Map::new();
                        for cell in Arc::try_unwrap(tls).ok().expect("workers joined") {
                            fold(&mut agg, &cell.into_inner());
                        }
                        black_box(agg.len());
                        d
                    }
                    // the slot is found once, like a mergex handle
                    "thread_local_hoisted" => {
                        let tls: Arc<ThreadLocal<RefCell<Map>>> = Arc::new(ThreadLocal::new());
                        let t2 = Arc::clone(&tls);
                        let d = timed(n, move |_| {
                            let slot = t2.get_or_default();
                            for i in 0..OPS {
                                *slot.borrow_mut().entry(i % KEYS).or_insert(0) += 1;
                            }
                        });
                        // the workers are joined, so this is the only handle left;
                        // consuming it avoids needing T: Sync just to aggregate
                        let mut agg = Map::new();
                        for cell in Arc::try_unwrap(tls).ok().expect("workers joined") {
                            fold(&mut agg, &cell.into_inner());
                        }
                        black_box(agg.len());
                        d
                    }
                    "thread_local_padded" => {
                        let tls: Arc<ThreadLocal<PaddedCell>> = Arc::new(ThreadLocal::new());
                        let t2 = Arc::clone(&tls);
                        let d = timed(n, move |_| {
                            let slot = t2.get_or_default();
                            for i in 0..OPS {
                                *slot.0.borrow_mut().entry(i % KEYS).or_insert(0) += 1;
                            }
                        });
                        let mut agg = Map::new();
                        for cell in Arc::try_unwrap(tls).ok().expect("workers joined") {
                            fold(&mut agg, &cell.0.into_inner());
                        }
                        black_box(agg.len());
                        d
                    }
                    _ => unreachable!(),
                };
                rates.push((OPS as f64 * n as f64) / dt.as_secs_f64() / 1e6);
            }
            row.push_str(&format!("{:>10.1}", median(rates)));
        }
        println!("{row}");
    }
}

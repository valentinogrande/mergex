//! Run-to-run repeatability for the concurrent write workload.
//!
//! criterion averages variance away inside one invocation; what we want here is
//! the spread ACROSS runs. Two things differ from the criterion benches:
//!
//!  * thread spawn and join sit outside the timed region. A barrier holds every
//!    worker on the start line, the clock starts, a second barrier releases when
//!    the last worker is done. Only the work is on the clock.
//!  * every configuration is run RUNS times and reported as median [min, max],
//!    so a figure can be quoted with the spread it actually has.

use left_right::{Absorb, WriteHandle};
use mergex::Mergex;
use std::hint::black_box;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

const OPS: i64 = 20_000;
const RUNS: usize = 15;
const THREADS: [usize; 8] = [1, 2, 4, 8, 12, 16, 24, 32];

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

/// Spawns `n` workers, holds them at the line, times only the work.
fn timed<W, F>(n: usize, make: W) -> Duration
where
    W: Fn(usize) -> F::Arg,
    F: Worker,
{
    let barrier = Arc::new(Barrier::new(n + 1));
    let hs: Vec<_> = (0..n)
        .map(|k| {
            let arg = make(k);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                F::run(arg);
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

trait Worker {
    type Arg: Send + 'static;
    fn run(arg: Self::Arg);
}

macro_rules! worker {
    ($name:ident, $arg:ty, $k:pat => $body:block) => {
        struct $name;
        impl Worker for $name {
            type Arg = $arg;
            fn run($k: Self::Arg) $body
        }
    };
}

worker!(WMergex, Mergex<i64>, son => {
    for i in 1..=OPS { son.update(|slot| *slot += i); }
});
worker!(WMutex, Arc<Mutex<i64>>, m => {
    for i in 1..=OPS { *m.lock().unwrap() += i; }
});
worker!(WRwLock, Arc<RwLock<i64>>, m => {
    for i in 1..=OPS { *m.write().unwrap() += i; }
});
worker!(WAtomic, Arc<AtomicI64>, a => {
    for i in 1..=OPS { a.fetch_add(i, Ordering::Relaxed); }
});
worker!(WShard, (Arc<Vec<Mutex<i64>>>, usize), (s, k) => {
    for i in 1..=OPS { *s[k].lock().unwrap() += i; }
});
worker!(WShardPad, (Arc<Vec<Padded>>, usize), (s, k) => {
    for i in 1..=OPS { *s[k].0.lock().unwrap() += i; }
});
worker!(WPlain, (), _u => {
    let mut local = 0i64;
    for i in 1..=OPS { local += i; black_box(&local); }
});
worker!(WLeftRight, Arc<Mutex<WriteHandle<Counter, Add>>>, w => {
    for i in 1..=OPS {
        let mut g = w.lock().unwrap();
        g.append(Add::By(i));
        if i % 1000 == 0 { g.publish(); }
    }
});

fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() {
    println!(
        "{} runs per configuration, {} ops per thread, spawn/join outside the clock\n",
        RUNS, OPS
    );
    println!(
        "{:<16}{:>7}{:>12}{:>12}{:>12}{:>9}",
        "structure", "thr", "median", "min", "max", "spread"
    );

    for threads in THREADS {
        for name in [
            "plain_local",
            "mergex",
            "sharded_padded",
            "sharded",
            "Atomic",
            "Mutex",
            "RwLock",
            "left_right",
        ] {
            let mut rates = Vec::with_capacity(RUNS);
            for _ in 0..RUNS {
                let dt = match name {
                    "plain_local" => timed::<_, WPlain>(threads, |_| ()),
                    "mergex" => {
                        let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
                        let d = timed::<_, WMergex>(threads, |_| root.copy());
                        black_box(root.get());
                        d
                    }
                    "sharded" => {
                        let s: Arc<Vec<Mutex<i64>>> =
                            Arc::new((0..threads).map(|_| Mutex::new(0)).collect());
                        timed::<_, WShard>(threads, |k| (Arc::clone(&s), k))
                    }
                    "sharded_padded" => {
                        let s: Arc<Vec<Padded>> =
                            Arc::new((0..threads).map(|_| Padded(Mutex::new(0))).collect());
                        timed::<_, WShardPad>(threads, |k| (Arc::clone(&s), k))
                    }
                    "Atomic" => {
                        let a = Arc::new(AtomicI64::new(0));
                        timed::<_, WAtomic>(threads, |_| Arc::clone(&a))
                    }
                    "Mutex" => {
                        let m = Arc::new(Mutex::new(0i64));
                        timed::<_, WMutex>(threads, |_| Arc::clone(&m))
                    }
                    "RwLock" => {
                        let m = Arc::new(RwLock::new(0i64));
                        timed::<_, WRwLock>(threads, |_| Arc::clone(&m))
                    }
                    "left_right" => {
                        let (w, r) = left_right::new_from_empty::<Counter, Add>(Counter(0));
                        let w = Arc::new(Mutex::new(w));
                        let d = timed::<_, WLeftRight>(threads, |_| Arc::clone(&w));
                        w.lock().unwrap().publish();
                        black_box(r.enter().map(|g| g.0).unwrap_or(0));
                        d
                    }
                    _ => unreachable!(),
                };
                rates.push((OPS as f64 * threads as f64) / dt.as_secs_f64() / 1e6);
            }
            let (med, min, max) = stats(rates);
            println!(
                "{:<16}{:>7}{:>12.1}{:>12.1}{:>12.1}{:>8.1}%",
                name,
                threads,
                med,
                min,
                max,
                (max - min) / med * 100.0
            );
        }
        println!();
    }
}

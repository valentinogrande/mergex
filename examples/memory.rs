//! Bytes actually allocated by each structure, measured with a counting
//! global allocator rather than estimated from struct sizes.

use mergex::Mergex;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

static LIVE: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new, Ordering::Relaxed);
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Live heap bytes held by whatever `build` returns, with the harness excluded.
fn measure<T>(build: impl FnOnce() -> T) -> usize {
    let before = LIVE.load(Ordering::Relaxed);
    let held = build();
    let after = LIVE.load(Ordering::Relaxed);
    drop(held);
    after - before
}

#[repr(align(128))]
struct Padded(#[allow(dead_code)] Mutex<i64>);

fn main() {
    println!("{:<18}{:>10}{:>10}{:>10}{:>12}", "structure", "n=8", "n=64", "n=512", "bytes/prod");

    for (name, f) in [
        (
            "mergex",
            &(|n: usize| {
                let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
                let sons: Vec<_> = (0..n).map(|_| root.copy()).collect();
                Box::new((root, sons)) as Box<dyn std::any::Any>
            }) as &dyn Fn(usize) -> Box<dyn std::any::Any>,
        ),
        (
            "Mutex (shared)",
            &(|_n: usize| Box::new(Arc::new(Mutex::new(0i64))) as Box<dyn std::any::Any>),
        ),
        (
            "RwLock (shared)",
            &(|_n: usize| Box::new(Arc::new(RwLock::new(0i64))) as Box<dyn std::any::Any>),
        ),
        (
            "Atomic (shared)",
            &(|_n: usize| Box::new(Arc::new(AtomicI64::new(0))) as Box<dyn std::any::Any>),
        ),
        (
            "sharded",
            &(|n: usize| {
                Box::new(Arc::new(
                    (0..n).map(|_| Mutex::new(0i64)).collect::<Vec<_>>(),
                )) as Box<dyn std::any::Any>
            }),
        ),
        (
            "sharded_padded",
            &(|n: usize| {
                Box::new(Arc::new(
                    (0..n).map(|_| Padded(Mutex::new(0i64))).collect::<Vec<_>>(),
                )) as Box<dyn std::any::Any>
            }),
        ),
        (
            "plain_local",
            &(|n: usize| Box::new(vec![0i64; n]) as Box<dyn std::any::Any>),
        ),
    ] {
        let sizes: Vec<usize> = [8usize, 64, 512].iter().map(|&n| measure(|| f(n))).collect();
        let per = (sizes[2] as f64 - sizes[0] as f64) / (512.0 - 8.0);
        println!(
            "{:<18}{:>10}{:>10}{:>10}{:>12.1}",
            name, sizes[0], sizes[1], sizes[2], per
        );
    }
}

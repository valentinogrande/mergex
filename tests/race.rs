//! Statistical race tests. They run real threads, so they are excluded
//! under `--cfg loom`; tests/loom.rs is the exhaustive counterpart.
#![cfg(not(loom))]

use mergex::Mergex;
use std::thread;

/// A writer pushing 1..=N with a max-merge, racing a merging reader.
/// After the writer joins, a final get() must report N: the last value written
/// cannot be dropped on the floor.
#[test]
fn the_last_set_is_never_swallowed() {
    const N: i64 = 2_000;
    let mut lost = 0u64;
    let mut worst = N;

    for _ in 0..300 {
        let root = Mergex::new(0i64, 0, |father: &mut i64, son: &i64| *father = (*father).max(*son));
        let son = root.copy();

        let s = son.clone();
        let t = thread::spawn(move || {
            for i in 1..=N {
                s.set(i);
            }
        });

        while !t.is_finished() {
            root.get(); // merging reader
        }
        t.join().unwrap();

        let final_value = root.get();
        if final_value != N {
            lost += 1;
            worst = worst.min(final_value);
        }
    }

    println!("corridas que perdieron el ultimo set: {lost}/300  (peor valor visto: {worst}, esperado {N})");
    assert_eq!(lost, 0);
}

/// Two readers merging while a writer runs. The subtree_dirty fast path lets one
/// reader bail out while the other is mid-merge, so an individual get() may come
/// back stale -- but nothing may be lost: once the writer has joined and the tree
/// is quiet, a final get() must be exact.
#[test]
fn concurrent_readers_never_lose_a_write() {
    const N: i64 = 5_000;

    for _ in 0..50 {
        let root = Mergex::new(0i64, 0, |father: &mut i64, son: &i64| *father = (*father).max(*son));
        let son = root.copy();

        let s = son.clone();
        let writer = thread::spawn(move || {
            for i in 1..=N {
                s.set(i);
            }
        });

        // second merging reader, bounded so it can never spin forever
        let r = root.clone();
        let reader = thread::spawn(move || {
            let mut seen = 0;
            for _ in 0..20_000 {
                seen = seen.max(r.get());
            }
            seen
        });

        while !writer.is_finished() {
            root.get();
        }
        writer.join().unwrap();
        let seen = reader.join().unwrap();

        assert!(seen <= N, "un reader vio {seen}, mas que el maximo escrito {N}");
        assert_eq!(root.get(), N, "se perdio el ultimo set");
    }
}

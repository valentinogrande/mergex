//! Exhaustive interleaving checks, run under the loom model checker.
//!
//!     RUSTFLAGS="--cfg loom" cargo test --release --test loom
//!
//! tests/race.rs covers the same ground statistically: it spins real threads a
//! few hundred times and checks nothing was dropped. That is worth much less
//! than it looks. It exercises one interleaving schedule, on one machine, under
//! one memory model -- and on x86 that model is TSO, which hands out
//! acquire/release on plain loads and stores for free. A missing `Ordering`
//! sails through it and then fails on aarch64. loom explores every interleaving
//! the C11 model permits, so it is the machine-independent version of the same
//! question.
//!
//! What is actually under test is the flag protocol. `mark_ancestors` walks
//! bottom-up and stops at the first ancestor already flagged:
//!
//! ```ignore
//! if flag.load(Ordering::Acquire) || flag.swap(true, Ordering::AcqRel) { return }
//! ```
//!
//! while `merge_sons` clears flags top-down. Those two directions cross, so the
//! state `grandson_flag == true && root_flag == false` is reachable in the
//! middle of a merge -- and in that state a writer bails out without ever
//! raising the root's flag. Nothing is lost only because `merge_sons` clears
//! its own flag *before* walking its sons: a writer that saw a flag still up
//! knows the merger has not descended past that point yet. That argument spans
//! three atomics and two directions of travel. These models are what keeps it
//! honest across refactors.
//!
//! Each model asserts only on the *final* read, after every writer has joined.
//! A `get()` racing a live writer is allowed to come back stale; that is the
//! documented contract. Losing a write that has already returned is not.
#![cfg(loom)]

use loom::thread;
use mergex::Mergex;

fn adder() -> Mergex<i64> {
    Mergex::new(0i64, 0, |father: &mut i64, son: &i64| *father += *son)
}

/// loom's search is exponential in the number of preemption points, and a
/// single `get()` over a two-level tree is already a dozen of them. The bound
/// keeps each model to seconds rather than hours; it is the standard tradeoff,
/// and bounded search still covers every schedule with up to N preemptions.
static RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn model(f: impl Fn() + Sync + Send + 'static) {
    RUNS.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = None;
    builder.check(move || {
        RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        f()
    });
    println!(
        "  schedules exploradas: {}",
        RUNS.load(std::sync::atomic::Ordering::Relaxed)
    );
}

/// root -> son. One writer, one merging reader.
/// The base case: `dirty` raised under the data lock, `subtree_dirty` raised
/// after it, against a reader that clears both in the other order.
#[test]
fn a_write_to_a_son_survives_a_concurrent_read() {
    model(|| {
        let root = adder();
        let son = root.copy();

        let reader = root.clone();
        let w = thread::spawn(move || son.set(1));
        let r = thread::spawn(move || {
            reader.get();
        });

        w.join().unwrap();
        r.join().unwrap();

        assert_eq!(root.get(), 1, "se perdio el set del hijo");
    });
}

/// root -> son -> grandson. One writer at the bottom, one merging reader at
/// the top.
///
/// This is the model that matters. Two levels of `ancestor_flags` means the
/// writer's bottom-up walk and the reader's top-down clear can cross in the
/// middle, which is the only place the early-return in `mark_ancestors` can be
/// wrong. A one-level tree cannot reach that state.
#[test]
fn a_write_to_a_grandson_survives_a_concurrent_read() {
    model(|| {
        let root = adder();
        let son = root.copy();
        let grandson = son.copy();

        let reader = root.clone();
        let w = thread::spawn(move || grandson.set(1));
        let r = thread::spawn(move || {
            reader.get();
        });

        w.join().unwrap();
        r.join().unwrap();

        assert_eq!(root.get(), 1, "se perdio el set del nieto");
    });
}

/// root -> son -> {g1, g2}. One thread writes both grandsons in sequence while
/// a reader merges.
///
/// The second `set` is the point: by then `son.subtree_dirty` is already up, so
/// `mark_ancestors` takes its early return and never touches the root's flag.
/// If the reader has meanwhile cleared the root's flag top-down, the root is
/// left believing its subtree is clean while a fresh write sits two levels
/// down. Only the clear-before-walk ordering saves it.
#[test]
fn the_early_return_in_mark_ancestors_never_hides_a_write() {
    model(|| {
        let root = adder();
        let son = root.copy();
        let g1 = son.copy();
        let g2 = son.copy();

        let reader = root.clone();
        let w = thread::spawn(move || {
            g1.set(1);
            g2.set(1); // sees son.subtree_dirty already up, bails out early
        });
        let r = thread::spawn(move || {
            reader.get();
        });

        w.join().unwrap();
        r.join().unwrap();

        assert_eq!(root.get(), 2, "el early-return se comio un set");
    });
}

/// Two writers on sibling sons, no reader. Whichever of them loses the race to
/// flag the father must not conclude the work is already done.
#[test]
fn two_writers_racing_to_flag_the_same_father() {
    model(|| {
        let root = adder();
        let a = root.copy();
        let b = root.copy();

        let ta = thread::spawn(move || a.set(1));
        let tb = thread::spawn(move || b.set(1));
        ta.join().unwrap();
        tb.join().unwrap();

        assert_eq!(root.get(), 2, "un hermano piso al otro");
    });
}

/// A writer that accumulates in place, against a reader that folds mid-stream.
///
/// The son keeps its running total across writes, so the reader has to take the
/// delta and leave the identity behind. If a fold cloned the value out instead,
/// a read landing between the two increments would fold `1`, and the next fold
/// would carry `2` -- the same increment twice, for a total of 3. Under loom
/// that interleaving is not a matter of luck: it is enumerated.
#[test]
fn an_interleaved_read_never_folds_the_same_delta_twice() {
    model(|| {
        let root = adder();
        let son = root.copy();

        let reader = root.clone();
        let w = thread::spawn(move || {
            son.update(|x| *x += 1);
            son.update(|x| *x += 1);
        });
        let r = thread::spawn(move || {
            reader.get();
        });

        w.join().unwrap();
        r.join().unwrap();

        assert_eq!(root.get(), 2, "se contó un delta dos veces");
    });
}

/// Two writes to the same grandson, where the second one skips the walk up
/// because the `dirty` bit was still raised, against a reader merging from the
/// root.
///
/// `update` only walks the ancestors on the false->true transition of `dirty`:
/// the walk takes a cache line shared by every writer in the subtree, and
/// paying for it once per fold instead of once per write is worth about ten
/// times the write throughput under eight threads. What makes the skip legal is
/// that a merge clears `dirty` under the same data lock the writer holds, so a
/// bit that is still up means no merge has folded this node since the write
/// that raised it. This model is what says so.
#[test]
fn a_second_write_that_skips_the_mark_is_still_found() {
    model(|| {
        let root = adder();
        let son = root.copy();
        let grandson = son.copy();

        let reader = root.clone();
        let w = thread::spawn(move || {
            grandson.update(|x| *x += 1); // raises dirty, walks up
            grandson.update(|x| *x += 1); // dirty still up -> no walk
        });
        let r = thread::spawn(move || {
            reader.get();
        });

        w.join().unwrap();
        r.join().unwrap();

        assert_eq!(root.get(), 2, "el segundo write se salteo la marca y se perdio");
    });
}

//! A tree of per-thread slots that are folded lazily into their father.
//!
//! Writers only ever touch the node they own, so they never share a cache line;
//! the aggregate is rebuilt on demand, when someone reads the root.
//!
//! # Contract
//!
//! `merge` and `identity` must form a commutative monoid over `T`:
//!
//!  * **identity**: `merge(&mut x, &identity)` leaves `x` unchanged.
//!  * **associative** and **commutative**: sons are folded in an order that
//!    depends on thread scheduling, so any order must give the same answer.
//!
//! A node's data is the **delta not yet folded into its father**, not a local
//! copy of the aggregate. A fresh son starts at `identity`, and a node is reset
//! to `identity` as soon as its data has been folded upward.

#![warn(missing_docs)]

use std::mem;

// The README's example is compiled and run as a doctest. It is the public API's
// only prose description of itself, and it has drifted out of date before -- a
// signature changed and the snippet kept claiming the old one.
#[cfg(all(doctest, not(loom)))]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;

// Under `--cfg loom` the locks and the flags are rebuilt on loom's instrumented
// primitives, so the model checker can drive every interleaving the memory
// model allows. See tests/loom.rs.
//
// `Arc`/`Weak` stay on std: loom 0.7 has no `Weak` and its `Arc` cannot hold an
// unsized `dyn Fn`, which is what `MergeFn` is. No loss for what is under test
// here -- the protocol lives in the AtomicBools and the Mutexes, not in the
// refcounts -- but it does mean loom is not watching the `Arc` graph, so it
// will not catch a leaked cycle.
#[cfg(loom)]
use loom::sync::Mutex;
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::Mutex;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Merge function: folds a son's data into its father's data.
/// Must be associative and commutative; see the crate docs.
pub type MergeFn<T> = Arc<dyn Fn(&mut T, &T) + Send + Sync>;

/// The shared node. Never handed out directly: users get a `Mergex` handle.
///
/// Aligned to two cache lines: nodes are per-writer, so two of them landing on
/// one line puts unrelated threads in a cache fight. Without this the write
/// benchmark is bimodal, swinging 3x on allocator luck alone.
#[repr(align(64))]
struct Node<T> {
    // There is deliberately no link back to the father. A dying node does not
    // push its delta upward -- it leaves it where it is, and the father, which
    // owns it, folds it on the next merge like any other son. That removes the
    // two ways a hand-written flush loses data: forgetting the pending delta,
    // and folding it from outside a merge walk, where `merge_sons` would not
    // propagate it any further.
    sons: Mutex<Vec<Arc<Node<T>>>>,
    // live `Mergex` handles pointing here. `Arc::strong_count` cannot stand in:
    // the father holds one, and `snapshot_sons` clones more for the duration of
    // a merge, so it never settles.
    handles: AtomicUsize,
    // set once the last handle is gone. The node keeps its place, and whatever
    // it still owes, until a merger folds it and reaps it.
    dead: AtomicBool,
    data: Mutex<T>,
    dirty: AtomicBool, // if true means that it is not sync with father node
    // conservative hint: true when some strict descendant may be dirty. Lets a
    // clean subtree be skipped in O(1) instead of being walked son by son.
    // Never a false negative, which is the only direction that would lose data.
    subtree_dirty: Arc<AtomicBool>,
    // the subtree_dirty flags of every ancestor, father first. Built once at
    // copy() time so a write never has to upgrade a Weak: doing that per set
    // is a CAS on the root's shared refcount from every writer at once.
    ancestor_flags: Vec<Arc<AtomicBool>>,
    // what a node holds when it has nothing pending for its father. Shared down
    // from the root like `merge`, so every node of a tree agrees on it.
    identity: Arc<T>,
    merge: MergeFn<T>,
}

/// Handle to a node in the tree. Cloning (or moving one into a thread) shares
/// the node, it does not copy it.
///
/// Except at the root, the data a node holds is the delta still owed to its
/// father, not the aggregate: it goes back to `identity` every time it is
/// folded upward.
pub struct Mergex<T> {
    node: Arc<Node<T>>,
}

impl<T> Clone for Mergex<T> {
    fn clone(&self) -> Self {
        self.node.handles.fetch_add(1, Ordering::Relaxed);
        Mergex {
            node: Arc::clone(&self.node),
        }
    }
}

impl<T> Drop for Mergex<T> {
    /// Dropping the last handle only plants a tombstone. It cannot unlink the
    /// node here: the node may still owe its father a delta, and only a merger
    /// -- walking down from above, inside its own fold -- is in a position to
    /// collect it. See `Node::reap`.
    fn drop(&mut self) {
        if self.node.handles.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.node.dead.store(true, Ordering::Release);
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mergex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sons: Vec<Mergex<T>> = self
            .node
            .sons
            .lock()
            .unwrap()
            .iter()
            .map(|n| Mergex { node: Arc::clone(n) })
            .collect();
        f.debug_struct("Mergex")
            .field("data", &*self.node.data.lock().unwrap())
            .field("dirty", &self.node.dirty.load(Ordering::Acquire))
            .field("sons", &sons)
            .finish()
    }
}

impl<T: Clone> Mergex<T> {
    /// Builds a root holding `data`.
    ///
    /// `identity` is the neutral element of `merge`: folding it into any value
    /// must leave that value unchanged. Every node is reset to it once its data
    /// has been folded into its father, which is what stops the same delta from
    /// being counted twice by two successive reads.
    pub fn new(data: T, identity: T, merge: impl Fn(&mut T, &T) + Send + Sync + 'static) -> Self {
        Mergex {
            node: Arc::new(Node {
                sons: Mutex::new(vec![]),
                handles: AtomicUsize::new(1),
                dead: AtomicBool::new(false),
                data: Mutex::new(data),
                dirty: AtomicBool::new(false),
                subtree_dirty: Arc::new(AtomicBool::new(false)),
                ancestor_flags: vec![],
                identity: Arc::new(identity),
                merge: Arc::new(merge),
            }),
        }
    }

    /// Registers a new son and returns its handle. Move it into a thread.
    ///
    /// The son starts at `identity`: it owes its father nothing yet. Seeding it
    /// with a value instead would be a write nobody ever asked for, and would
    /// land in the aggregate the first time any of its own sons was folded.
    pub fn copy(&self) -> Self {
        let son = Arc::new(Node {
            sons: Mutex::new(vec![]),
            handles: AtomicUsize::new(1),
            dead: AtomicBool::new(false),
            data: Mutex::new((*self.node.identity).clone()),
            dirty: AtomicBool::new(false),
            subtree_dirty: Arc::new(AtomicBool::new(false)),
            ancestor_flags: {
                let mut f = Vec::with_capacity(self.node.ancestor_flags.len() + 1);
                f.push(Arc::clone(&self.node.subtree_dirty));
                f.extend(self.node.ancestor_flags.iter().cloned());
                f
            },
            identity: Arc::clone(&self.node.identity),
            merge: Arc::clone(&self.node.merge),
        });
        self.node.sons.lock().unwrap().push(Arc::clone(&son));
        Mergex { node: son }
    }

    /// Overwrites this node's data and marks it out of sync with its father.
    pub fn set(&self, data: T) {
        self.update(|slot| *slot = data);
    }

    /// Read-modify-writes this node's data under a single lock and marks it out
    /// of sync with its father. `set` is this with a closure that overwrites.
    pub fn update(&self, f: impl FnOnce(&mut T)) {
        // the dirty flag is raised while still holding the data lock, so a
        // concurrent merge can never clear it in between and swallow the write
        let mut guard = self.node.data.lock().unwrap();
        f(&mut guard);
        // A plain load and store, not a read-modify-write: `dirty` is only ever
        // *written* under this data lock -- by `update` here, and by the merge
        // that clears it -- so holding the lock already makes the pair atomic
        // against everyone who could change it. The one unlocked reader,
        // `merge_sons`' fast path, only loads. The old value says whether this
        // node already owed its father something.
        let was_dirty = self.node.dirty.load(Ordering::Relaxed);
        self.node.dirty.store(true, Ordering::Release);
        drop(guard);

        // Walking up is what costs: `mark_ancestors` takes a line shared by
        // every writer in the subtree, exclusively. Skipping it when the bit was
        // already up turns a burst of writes on one node into a single walk,
        // which is the difference between a per-write shared RMW and a
        // per-fold one -- roughly a factor of ten under eight writers.
        //
        // Sound because a merge clears `dirty` under this same data lock. So
        // `was_dirty` true means no merge has folded this node since the write
        // that raised the bit, and that write is the one that flagged the
        // ancestors: either it has finished doing so, or it is still on its way
        // up, and its RMWs land in the same modification order the merger's
        // clearing swap reads from. Nothing is left unflagged that a merge has
        // already walked past.
        if !was_dirty {
            self.node.mark_ancestors();
        }
    }

    /// This node's own data, with no merging. Away from the root that is the
    /// delta still owed to the father, so it reads back as `identity` right
    /// after a fold, not as everything this node has ever written.
    pub fn get_threaded(&self) -> T {
        self.node.data.lock().unwrap().clone()
    }

    /// Folds every dirty descendant into this node, deepest first, and returns
    /// the result. Called on the root, that is the aggregate.
    ///
    /// When nothing below has changed this is a single atomic load, whatever
    /// the size of the subtree. When several threads call it at once, one wins
    /// the sweep and the others return what is currently there rather than
    /// waiting -- so an individual call can be stale under concurrent readers,
    /// though nothing is lost and a call on a quiet tree is exact.
    ///
    /// ```
    /// # use mergex::Mergex;
    /// let root = Mergex::new(10i64, 0, |f: &mut i64, s: &i64| *f += *s);
    /// let son = root.copy();
    /// son.set(5);
    /// assert_eq!(root.get(), 15);
    /// assert_eq!(root.get(), 15); // the delta is not folded twice
    /// ```
    pub fn get(&self) -> T {
        self.node.merge_sons();
        self.get_threaded()
    }

    /// Whether any descendant has something pending. `false` does not promise
    /// that a `get` is free, only that no write is waiting to be folded.
    ///
    /// Costs one atomic load on a clean subtree, whatever its size.
    pub fn check_children(&self) -> bool {
        self.node.check_children()
    }

    /// Whether this node itself owes its father a delta.
    pub fn check_dirty_bit(&self) -> bool {
        self.node.dirty.load(Ordering::Acquire)
    }
}

impl<T: Clone> Node<T> {
    /// Flags every ancestor as having a dirty descendant. Stops at the first
    /// one that was already flagged, so a burst of sets on the same node costs
    /// one RMW, not one walk per set.
    fn mark_ancestors(&self) {
        for flag in &self.ancestor_flags {
            // Always a read-modify-write, never a `load` fast path, even though
            // the flag is usually already up and the RMW takes the line
            // exclusive. The swap is not here to change the flag -- it is here
            // to *publish*.
            //
            // `merge_sons` reads a son's `dirty` bit without taking that son's
            // data lock, so the only thing that makes the raised bit visible to
            // it is a release edge on the flag it clears. A `load` that returns
            // true and bails out releases nothing: the writer leaves having
            // published no edge at all, the merger may then read `dirty` stale,
            // skip the son, and clear the flags on its way out -- and the write
            // is gone for good, because nothing is left flagged to bring anyone
            // back. loom finds it in a two-level tree; see tests/loom.rs.
            //
            // A plain load is also unsound for a second reason, one the lock
            // would not fix either: it can read a stale `true` for an ancestor a
            // merge has just cleared, and then the walk stops short of the root
            // and no later read ever comes down. An RMW cannot -- it always
            // reads the latest value in the flag's modification order.
            //
            // The RMW is what closes it: it lands in the flag's modification
            // order, so the merger's own `swap(false)` either reads it -- and
            // then acquires everything this writer did, `dirty` included -- or
            // precedes it, in which case the flag is left up and the next read
            // comes back down.
            if flag.swap(true, Ordering::AcqRel) {
                return;
            }
        }
    }

    /// Folds every dirty descendant into this node, deepest first.
    /// Returns whether `self.data` changed.
    fn merge_sons(&self) -> bool {
        // cleared before the walk, not after: a set landing mid-walk re-flags
        // this node and is picked up by the next merge instead of being lost
        if !self.subtree_dirty.swap(false, Ordering::AcqRel) {
            return false;
        }

        let mut changed = false;
        let mut spent: Vec<Arc<Node<T>>> = Vec::new();
        // one clone of the handle list per level: the sons lock must not be held
        // across the recursion or copy() would block for the whole subtree merge.
        // The subtree_dirty check above means a clean tree never gets here.
        for son in self.snapshot_sons() {
            // grandsons first: a son that absorbs a grandson is itself out of sync
            let son_changed = son.merge_sons();
            // cheap check first, to skip the lock on a clean son
            if !son.dirty.load(Ordering::Acquire) && !son_changed {
                // Nothing to fold. Reaping rides along on this same walk rather
                // than a second pass: a second `snapshot_sons` would allocate a
                // vector and touch every refcount again on every merge, spent
                // sons or not, and that showed up as +47% on a 1000-son fold.
                if son.is_spent() {
                    spent.push(son);
                }
                continue;
            }
            // read and clear under the son's lock: this pairs with set(), so the
            // two can never interleave. copy the value out before taking our own
            // lock, we must never hold two at once.
            let son_data = {
                let mut guard = son.data.lock().unwrap();
                son.dirty.store(false, Ordering::Release);
                // take the delta and leave the identity behind. Cloning it out
                // and leaving it in place would fold the same delta again on
                // the next read: `update(|x| *x += 1)` twice, with a get() in
                // between, would land as +1 then +2.
                mem::replace(&mut *guard, (*son.identity).clone())
            };
            (self.merge)(&mut self.data.lock().unwrap(), &son_data);
            changed = true;
            // folded and empty-handed: this is the moment it becomes reapable
            if son.is_spent() {
                spent.push(son);
            }
        }
        if !spent.is_empty() {
            self.reap(&spent);
        }
        changed
    }

    /// A node with no handles left, nothing pending, and nothing underneath it.
    /// Anything less and unlinking it would lose data or orphan a live writer.
    fn is_spent(&self) -> bool {
        self.dead.load(Ordering::Acquire)
            && self.handles.load(Ordering::Acquire) == 0
            && !self.dirty.load(Ordering::Acquire)
            && self.sons.lock().unwrap().is_empty()
    }

    /// Unlinks spent sons. `swap_remove` semantics via `retain`: nothing caches
    /// a position, so there are no indices to patch. Because a dead son is
    /// removed rather than tombstoned in place, the list stays the length of the
    /// *live* sons, which is what keeps `snapshot_sons` from growing with the
    /// number of nodes the process has ever created.
    fn reap(&self, spent: &[Arc<Node<T>>]) {
        // Two regimes. A pool recycling one worker at a time retires a single
        // son per merge, and there a linear scan beats allocating and sorting.
        // A whole generation retiring at once makes `spent` as long as `sons`,
        // and then `any(ptr_eq)` inside `retain` is quadratic -- 362us to reap a
        // thousand sons, against 154us by address.
        const LINEAR: usize = 8;
        let marks: Vec<*const Node<T>> = if spent.len() > LINEAR {
            let mut m: Vec<*const Node<T>> = spent.iter().map(Arc::as_ptr).collect();
            m.sort_unstable();
            m
        } else {
            Vec::new()
        };
        let is_marked = |s: &Arc<Node<T>>| {
            if spent.len() > LINEAR {
                marks.binary_search(&Arc::as_ptr(s)).is_ok()
            } else {
                spent.iter().any(|d| Arc::ptr_eq(s, d))
            }
        };

        let mut sons = self.sons.lock().unwrap();
        sons.retain(|s| {
            if !is_marked(s) {
                return true;
            }
            // Re-check the flags, but never the son's own `sons` list: taking
            // that lock here would mean holding two at once, which this crate
            // does not do anywhere else. It is not needed. A new son can only
            // appear through `copy()`, which needs a handle, and this node has
            // none and can never be handed one again -- `dead` is permanent
            // because nothing hands out a handle to an existing node.
            s.handles.load(Ordering::Acquire) != 0 || s.dirty.load(Ordering::Acquire)
        });
    }

    fn check_children(&self) -> bool {
        if !self.subtree_dirty.load(Ordering::Acquire) {
            return false;
        }
        self.snapshot_sons()
            .iter()
            .any(|s| s.dirty.load(Ordering::Acquire) || s.check_children())
    }

    /// Clones the son handles out so the `sons` lock is not held while
    /// recursing: another thread calling copy() on us would block on it.
    fn snapshot_sons(&self) -> Vec<Arc<Node<T>>> {
        self.sons.lock().unwrap().to_vec()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::thread;

    fn sum() -> Mergex<i32> {
        Mergex::new(7, 0, |father: &mut i32, son: &i32| *father += *son)
    }

    #[test]
    fn thread_updates_reach_the_root() {
        let root = sum();
        let t1 = root.copy();
        let t2 = root.copy();

        let a = thread::spawn(move || t1.set(10));
        let b = thread::spawn(move || t2.set(20));
        a.join().unwrap();
        b.join().unwrap();

        assert!(root.check_children());
        assert_eq!(root.get(), 37); // 7 + 10 + 20
        assert!(!root.check_children()); // dirty bits cleared
    }

    #[test]
    fn grandsons_merge_bottom_up() {
        let root = sum();
        let son = root.copy();
        let grandson = son.copy();
        grandson.set(5);

        assert!(root.check_children()); // seen through the son
        assert_eq!(root.get(), 12); // 7 + (0 + 5)
    }

    #[test]
    fn sons_created_inside_a_thread_register() {
        let root = sum();
        let t1 = root.copy();

        thread::spawn(move || {
            let grandson = t1.copy();
            grandson.set(1000);
        })
        .join()
        .unwrap();

        assert_eq!(root.get(), 1007); // 7 + (0 + 1000)
    }

    #[test]
    fn dropping_the_root_strands_the_son() {
        let son = {
            let root = sum();
            let son = root.copy();
            son.set(3); // never folded: the root is about to go away
            son
        }; // root handle dropped here, nothing else holds it

        // the tree is owned from the root down, so the son survives on its own
        // handle but has nowhere left to fold into
        assert_eq!(son.get_threaded(), 3);
    }

    #[test]
    fn a_dropped_handle_still_delivers_its_write() {
        let root = sum();
        {
            let worker = root.copy();
            worker.set(5);
        } // the worker is gone before anyone reads

        assert_eq!(root.get(), 12); // 7 + 5, collected by the merge
    }

    #[test]
    fn a_spent_son_is_unlinked_by_the_next_merge() {
        let root = sum();
        {
            let worker = root.copy();
            worker.set(5);
        }
        assert_eq!(root.node.sons.lock().unwrap().len(), 1); // still linked

        assert_eq!(root.get(), 12); // folds, then reaps
        assert_eq!(root.node.sons.lock().unwrap().len(), 0);
    }

    #[test]
    fn a_live_son_is_never_unlinked() {
        let root = sum();
        let worker = root.copy();
        worker.set(5);

        assert_eq!(root.get(), 12);
        assert_eq!(root.node.sons.lock().unwrap().len(), 1); // handle still held

        worker.set(2);
        assert_eq!(root.get(), 14); // and still usable
    }

    #[test]
    fn a_dead_son_holding_a_live_grandson_stays() {
        let root = sum();
        let grandson = {
            let son = root.copy();
            son.copy()
        }; // the intermediate handle is gone, but its son is not
        grandson.set(9);

        assert_eq!(root.get(), 16); // 7 + (0 + 9), through the dead intermediate
        assert_eq!(root.node.sons.lock().unwrap().len(), 1); // not reaped: not empty
    }

    #[test]
    fn a_pool_of_short_lived_workers_does_not_grow_the_son_list() {
        let root = Mergex::new(0i64, 0, |f: &mut i64, s: &i64| *f += *s);
        for i in 1..=100 {
            let w = root.copy();
            w.set(i);
            drop(w);
            root.get(); // a pool reads between tasks
        }

        assert_eq!(root.get(), (1..=100).sum::<i64>());
        assert_eq!(root.node.sons.lock().unwrap().len(), 0);
    }

    #[test]
    fn many_threads_registering_and_setting_at_once() {
        let root = Mergex::new(0i64, 0, |father: &mut i64, son: &i64| *father += *son);

        let handles: Vec<_> = (1..=50)
            .map(|i| {
                let r = root.clone(); // concurrent copy(): stresses the sons lock
                thread::spawn(move || r.copy().set(i))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(root.get(), (1..=50).sum::<i64>()); // 1275, nothing lost
    }

    #[test]
    fn merging_is_idempotent() {
        let root = sum();
        root.copy().set(10);

        assert_eq!(root.get(), 17);
        assert_eq!(root.get(), 17); // dirty bit cleared, no double counting
    }

    #[test]
    fn a_read_does_not_replay_an_earlier_delta() {
        let root = sum();
        let son = root.copy();

        // the son accumulates in place, so its data still holds the previous
        // increments. Each read must fold only what is new since the last one.
        son.update(|x| *x += 1);
        assert_eq!(root.get(), 8);
        son.update(|x| *x += 1);
        assert_eq!(root.get(), 9); // not 10: the first +1 is not folded twice
        son.update(|x| *x += 1);
        assert_eq!(root.get(), 10);
    }

    #[test]
    fn folding_resets_the_son_to_the_identity() {
        let root = sum();
        let son = root.copy();

        assert_eq!(son.get_threaded(), 0); // a fresh son owes its father nothing
        son.set(5);
        assert_eq!(root.get(), 12);
        assert_eq!(son.get_threaded(), 0); // and owes nothing again once folded
    }

    #[test]
    fn an_intermediate_node_is_reset_too() {
        let root = sum();
        let son = root.copy();
        let grandson = son.copy();

        grandson.set(4);
        assert_eq!(root.get(), 11);
        assert_eq!(son.get_threaded(), 0); // the son passed the delta on
        assert_eq!(grandson.get_threaded(), 0);

        grandson.set(4);
        assert_eq!(root.get(), 15); // 11 + 4, not 11 + 8
    }
}

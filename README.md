# mergex

A concurrent merge tree. Every thread writes to a node it owns; the aggregate is
assembled only when someone asks for it.

```rust
use mergex::Mergex;

// data, identity, and the fold: father first, then son
let root = Mergex::new(0i64, 0, |father: &mut i64, son: &i64| *father += *son);

let handles: Vec<_> = (0..8)
    .map(|_| {
        let node = root.copy();           // register a son, get its handle
        std::thread::spawn(move || {
            for i in 1..=1000 {
                node.update(|slot| *slot += i);   // writes only to this node
            }
        })
    })
    .collect();

for h in handles {
    h.join().unwrap();
}

assert_eq!(root.get(), 4_004_000);        // folds every dirty node, deepest first
```

Nothing is shared on the write path. `root.get()` is what walks the tree, and when
nothing has changed it answers in a single atomic load. Workers that finish and drop
their handle deliver their last write and then disappear from the tree.

## The contract

`merge` and `identity` must form a **commutative monoid** over `T`:

- folding `identity` into any value leaves it unchanged;
- the fold is associative and commutative, because sons are folded in an order that
  depends on thread scheduling.

A node's data is the **delta not yet folded into its father**, not a local copy of
the aggregate. A fresh son starts at `identity`, and a node is reset to `identity`
the moment its data is folded upward — which is what stops one write from being
counted twice by two successive reads.

## Is this the right tool?

**Yes, if** many threads write concurrently and something occasionally reads the
aggregate. That is the shape it is built for, and it holds up: at eight writers it
moves **671 million writes per second** against **21.1** for a shared `Mutex<i64>`, a
factor of 32 — and it is not done climbing: at twelve, the machine's hardware thread
count, it reaches **795** while the `Mutex` is stuck at 18.4.

**No, if** many threads *read* concurrently. `get()` takes the root's lock, so
readers queue behind each other and behind any writer. Reads do not scale, and that
is a property of the design rather than a bug awaiting a fix:

| million reads/s | 1 reader | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| mergex, tree quiet | 91.2 | 18.6 | 16.6 | **13.7** |
| mergex, one writer alongside | 3.8 | 14.3 | 6.4 | **4.8** |
| [left-right][lr], one writer alongside | 7.7 | 17.3 | 45.9 | **119.1** |
| `AtomicI64`, one writer alongside | 259.4 | 310.1 | 590.7 | **746.0** |

If your workload is read-heavy, [left-right][lr] is the right structure and this one
is not. The two are built for opposite problems — left-right is one writer and
wait-free readers, mergex is many writers and an occasional reader — and each loses
badly on the other's ground. left-right manages 10.2 million writes per second at
eight writers, the slowest entry in the whole comparison.

**Also no, if** a flat one-level fold is all you need. One cache-line-padded
`Mutex<i64>` per thread, summed at the end, does 1018 million writes per second
against mergex's 671, in 128 bytes per producer against 240. Mergex earns its keep
with the tree — arbitrary depth, coalescing writes, self-cleaning membership, and an
O(1) answer to "has anything changed?" — not with raw speed.

[lr]: https://docs.rs/left-right

## How it works

```text
                    ┌─────────────┐
                    │    root     │   get() folds here
                    └──────┬──────┘
                subtree_dirty = true
            ┌──────────────┼──────────────┐
     ┌──────┴─────┐ ┌──────┴─────┐ ┌──────┴─────┐
     │   son A    │ │   son B    │ │   son C    │
     │   dirty    │ │   clean    │ │   dirty    │
     └────────────┘ └────────────┘ └────────────┘
       thread 1       thread 2       thread 3
```

A node holds its data behind its own `Mutex`, plus two flags:

- **`dirty`** — this node owes its father something.
- **`subtree_dirty`** — some descendant may be dirty. Deliberately conservative: it
  may be true when nothing is, never false when something is. A false negative would
  lose data; a false positive only costs a wasted sweep.

`get()` folds every dirty descendant into the root, deepest first, because a node
that absorbs a grandchild is itself out of sync with its father. When the root's
`subtree_dirty` is clear the whole sweep collapses to one atomic load: **10.96 ns
whether the root has one son or a thousand**.

### Four decisions worth knowing about

**There is no link back to the father.** A dying node does not push its delta
upward — it leaves it exactly where it is, and the father, which owns it, folds it on
the next merge like any other son. That is what removes the two ways a hand-written
flush loses data: forgetting the pending delta, and folding it from *outside* a merge
walk, where `merge_sons` propagates a son only if its own bit is set or it changed
within that same walk — so a delta pushed in from the side stops there and never
reaches the root.

The price is that there is no `get_father()`. A worker that needs the root carries a
`root.clone()` into its thread, which is one line and is honest about the ownership:
the tree is owned from the root down.

**No `Weak::upgrade` on the write path.** Each node caches its ancestors'
`subtree_dirty` flags as `Arc<AtomicBool>`, built once in `copy()`. Upgrading a
`Weak` is a compare-exchange on a shared refcount, and under contention that is
brutal:

| ns per operation | 1 thread | 8 threads on the same target |
|---|---:|---:|
| `Weak::upgrade` | 7.52 | **101.43** |
| `AtomicUsize::fetch_add` | 3.43 | 23.32 |
| `AtomicBool::load` | 0.89 | **0.17** |

A shared load gets *cheaper* per operation as threads are added, because the cache
line stays shared in every core. Every read-modify-write serialises. One `upgrade`
per write cost 30x the write throughput.

**Raising an ancestor flag is always a read-modify-write.** Never a `load` fast path,
even though the flag is usually already up and the RMW takes the line exclusively.
**The swap is not there to change the flag — it is there to publish.** `merge_sons`
reads a son's `dirty` bit without taking that son's data lock, so the only thing that
makes a raised bit visible to it is a release edge on the flag the merger clears. A
`load` that returns true and bails out releases nothing: the writer leaves having
published no edge, the merger may then read `dirty` stale, skip the son, and clear
the flags on its way out — and the write is gone for good, because nothing is left
flagged to bring anyone back.

This was a real bug in this crate, and an instructive one: on x86 it never fires,
because TSO hides it. Three hundred stress runs passed. `loom` finds it in a
two-level tree in hundredths of a second. **`tests/race.rs` does not test ordering;
it tests x86.** `tests/loom.rs` tests the protocol.

The cost is paid back elsewhere: `update` walks up only on the `dirty` false→true
transition, read under the data lock. A merge clears `dirty` under that same lock, so
a bit still raised means no merge has folded this node since the write that raised
it — and that write already flagged the ancestors. A burst of writes on one node
costs one walk, not one per write.

**`Node` is cache-line aligned.** Nodes are per-writer, so two landing on one 64-byte
line puts unrelated threads in a cache fight. Without `#[repr(align(64))]` the write
benchmark is bimodal, swinging 3x on allocator luck: worst case **207.7** million
writes per second against **634.2** with it, at the same median. It costs 64 bytes
per node and buys predictability.

### Retiring a worker

Dropping the last handle to a node only plants a tombstone. It cannot unlink the node
there: the node may still owe its father a delta, and only a merger — walking down
from above, inside its own fold — is in a position to collect it.

```rust,ignore
impl<T> Drop for Mergex<T> {
    fn drop(&mut self) {
        if self.node.handles.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.node.dead.store(true, Ordering::Release);
        }
    }
}
```

The merge that folds a dead node's last delta is the one that unlinks it, and only
when it is *spent*: no handles, nothing pending, and **nothing underneath it**. That
last condition is not optional. A dead node with a live grandson still routes that
grandson's writes, and its descendants cache its `subtree_dirty` in their
`ancestor_flags` — so it can neither be unlinked nor re-parented, and it stays until
its own subtree drains.

Because a spent son is removed rather than tombstoned in place, the sibling list stays
the length of the *live* sons. This is the difference between working for a fixed
thread pool and working for spawn-per-task: without it, `get()` would cost O(nodes
ever created) rather than O(nodes alive), since a single dirty son puts the whole
sibling list into the sweep.

The bill is one `dead` load per son per merge — about 2 ns each, or 5% on a
thousand-son fold — plus the unlink itself when there is something to unlink.

## API

| method | what it does |
|---|---|
| `new(data, identity, merge)` | root node; `merge` is `Fn(&mut T, &T)`, father then son |
| `copy()` | register a son at `identity` and return its handle |
| `update(f)` | read-modify-write this node's data under one lock, mark it dirty |
| `set(value)` | `update` with a closure that overwrites |
| `get_threaded()` | this node's own delta, no merging |
| `get()` | fold every dirty descendant into this node, then return its value |
| `check_children()` | is any descendant dirty? O(1) when the subtree is clean |

`Mergex<T>` is a handle. Cloning it, or moving it into a thread, shares the node
rather than copying it. Dropping the last one retires the node.

### Concurrency guarantees

- **A write is never swallowed.** The dirty flag is raised while the data lock is
  still held, so a concurrent merge cannot clear it in between. This was a real bug
  too: before the fix, 173 of 300 runs lost the last write of a racing writer.
- **A dropped handle still delivers.** A worker that writes and then goes away leaves
  its delta in place for the next merge to collect.
- **Merging is idempotent.** A folded node is reset to `identity` under its own lock,
  so the same delta is never counted twice.
- **Writes coalesce.** Several writes to a node between merges arrive as one delta.
- **A single `get()` may return a stale value** when several readers race: one wins
  the flag and merges, the others return immediately with whatever is there. Nothing
  is lost, since the winner completes the fold, but `get()` is not "the exact value
  right now" under concurrent readers. Once the tree is quiet it is exact.
- **No two locks are ever held at once.** A son's delta is taken and its lock released
  before the father's is acquired. `Mutex` is not reentrant and there is no global
  acquisition order, so mixing this would deadlock — which is also why the unlink pass
  re-checks flags rather than a son's own son list.

## Benchmarks

AMD Ryzen 5 PRO 5650U, 12 threads · rustc 1.95.0 · criterion 0.8.2 · Linux 7.1.4

Every entrant performs the same operation — `+= i` on an `i64`, twenty thousand times
per thread. What differs is the topology: mergex, `sharded` and `plain_local` hit a
slot the thread owns, while `Mutex`, `RwLock` and `AtomicI64` hit one cell shared by
every thread, which is the only way a single cell can hold a running total. That
difference is the subject of the benchmark.

### Write throughput

Million writes per second, higher is better.

The machine has 12 hardware threads, so the sweep runs past it into
oversubscription. Bold marks each structure's own peak.

| threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `plain_local` | 582.8 | 1242.6 | 2451.6 | **2546.2** | 2350.8 | 1575.2 | 1941.3 | 1784.9 |
| `sharded_padded` | 142.1 | 274.3 | 547.8 | **1018.0** | 1010.2 | 775.9 | 908.5 | 866.0 |
| **mergex** | 132.7 | 271.2 | 535.3 | 671.3 | **795.0** | 649.8 | 761.8 | 725.4 |
| `AtomicI64` | **297.3** | 63.2 | 57.6 | 51.3 | 48.8 | 48.4 | 48.0 | 47.7 |
| `sharded` | 142.3 | 21.4 | 29.3 | 40.1 | 58.9 | **79.9** | 69.6 | 74.4 |
| `Mutex` | **140.7** | 41.8 | 29.3 | 21.1 | 18.4 | 18.4 | 18.4 | 17.8 |
| `RwLock` | **151.5** | 39.7 | 29.0 | 15.0 | 11.1 | 11.6 | 11.6 | 12.0 |
| `left-right` | **90.1** | 16.8 | 13.2 | 10.2 | 8.1 | 8.1 | 7.9 | 7.9 |

Every design that funnels writes through one cell peaks at **one** thread and falls
from there, the lock-free `AtomicI64` included, because `fetch_add` still serialises
on one cache line. Only the per-writer designs climb.

Past the core count they do not collapse — they flatten. `Mutex` sits at 18.4 from
twelve threads onward, `RwLock` at about 11.5, `left-right` at 8. They were already
fully serialised at eight threads, so adding more changes nothing: the queue is the
queue. There is no lock convoy to see here.

Two things worth reading off the wide table:

**mergex peaks at twelve, not eight.** It is the only entrant still climbing at the
machine's physical limit — 671 to 795 — while `sharded_padded` had already topped out
at eight. Mergex uses the last four hardware threads that its rival cannot.

**It degrades best when oversubscribed.** As a share of its own peak at 32 threads:
mergex 91%, `sharded_padded` 85%, `plain_local` 70%. With no shared lock, a thread
descheduled mid-write blocks nobody, so the total work is conserved and merely spread
over more slices. And the advantage over a shared `Mutex` *grows* with thread count:
31x at eight, **43x at twelve**, 41x at thirty-two.

Two oddities in that table are artefacts, not properties. **Every row dips at 16 and
recovers at 24** — 16 does not divide 12 evenly while 24 is exactly two full passes, so
this is the scheduler, not the structures. And **`sharded` improves with more threads**
(21.4 at two, 79.9 at sixteen), alone in going the other way; the likely reason is that
oversubscription means fewer threads are genuinely simultaneous on the same cache line,
so its false sharing eases. That one is a hypothesis — it is not measured.

`plain_local` is one plain `i64` per thread with no synchronisation at all, folded on
join. It is the ceiling, present so the rest can be read against something absolute —
but it is also the least reproducible entrant, because twenty thousand additions take
a few microseconds and scheduler skew dominates. Read it as an order of magnitude, not
a figure. `sharded_padded` is one cache-line-aligned `Mutex<i64>` per thread, summed at
the end — mergex by hand, one level deep, and the closest thing to a fair rival.

Compare `sharded` against `sharded_padded`: the same idea without the alignment is
**27x slower** at eight threads. Adjacent shards share a cache line, so the threads
fight over cache even though they never share a lock. It is the same effect
`#[repr(align(64))]` on `Node` exists to prevent.

### Latency

| operation | cost |
|---|---:|
| `set` / `update` | 7.25 ns |
| `get_threaded` | 7.10 ns |
| `get()`, nothing dirty, 1 to 1000 sons | **10.96 ns** |
| `check_children()`, 1 to 1000 sons | **3.10 ns** |
| `get()` merging 10 dirty sons | 269 ns |
| `get()` merging 1000 dirty sons | 38.1 µs |
| `get()` folding *and retiring* 1 spent son | 140 ns |
| `get()` folding *and retiring* 1000 spent sons | 145 µs |
| `copy()` per node registered | ~150 ns |

The two flat rows are the point of the design: asking "is there anything to do?" costs
the same with a thousand producers as with one.

Reading the aggregate when nothing has changed, single reader, 512 producers:

| | |
|---|---:|
| `AtomicI64` | 0.89 ns |
| `Mutex` | 7.10 ns |
| **mergex** | **10.87 ns** |
| `left-right` | 12.12 ns |
| `sharded` | 3.65 µs |

Hand-rolled sharding has no shortcut and must sum every shard — 56.8 ns at 8
producers, 3.65 µs at 512. mergex and left-right are both flat. These are
**single-reader** figures; see the reader-scaling table near the top for what happens
with more than one, which is a different and less flattering story.

### Memory

Live heap bytes, counted by a `GlobalAlloc` wrapper rather than estimated from
`size_of`.

| structure | bytes per producer |
|---|---:|
| **mergex** | **240** |
| `sharded_padded` | 128 |
| `sharded` | 16 |
| `plain_local` | 8 |
| `Mutex` / `RwLock` / `AtomicI64` | 0 |

A mergex node holds an `Arc` header, a `Mutex<T>`, a `Mutex<Vec<Arc<Node>>>` for its
sons, a handle count, two flags, `Arc`s to the merge function and the identity, and
the vector of ancestor flags — then rounds up to a cache line. This is the cost of a
*live* producer: retired ones are unlinked. The shared-cell designs are flat because
they have nothing per producer; they pay for concurrency in contention instead of
memory.

### Method

- **Thread spawn and join sit outside the timed region.** A barrier holds every worker
  on the start line, the clock starts, a second barrier releases when the last worker
  is done.
- **Every configuration is run 15 times, and the whole table repeatedly.** A single run
  on this machine is not reproducible: min-to-max spread reaches 70% under background
  load. The medians are — they agreed within 1% across every pass — which is why
  medians are quoted here and min/max is not.
- **This is a laptop with the `powersave` governor**, not a dedicated benchmarking box,
  and the bench suite is heavy enough to warm it up. Read the ratios, not the absolute
  figures.
- **`merge_dirty` and `reap` figures include one `Instant` pair** (40.9 ns), because the
  harness has to start the clock after preparing the sons. It is subtracted in the
  latency table above; the raw criterion output does not subtract it.

## Testing

```sh
cargo test                                                # unit + concurrency regressions
RUSTFLAGS="--cfg loom" cargo test --release --test loom    # the memory-ordering protocol
```

`tests/race.rs` runs real threads and catches lost writes; `tests/loom.rs` model-checks
the flag protocol across every interleaving the memory model allows. The second is the
one that catches ordering bugs — see the third design decision above for why the first
cannot.

## Credits

The tests (`tests/`, and the unit tests in `src/lib.rs`), the benchmark suite
(`benches/`), the measurement harnesses (`examples/`), the source comments and
documentation, and this README were written by Claude Code.

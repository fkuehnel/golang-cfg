# Conclusions: why plain iteration wins, and what real CFGs taught us

*Closing summary, 2026-08-26. The evidence for every claim here is a row in
[STATUS.md](STATUS.md) with its provenance; the raw data lives under
`vmbench/results/`; the mechanism model and its witnesses live under
`wolfram/`. This document is the narrative those tables cannot tell.*

## The finding

**Real-world control-flow graphs are shaped so that plain postorder iteration
is already near-optimal for liveness, and no amount of algorithmic
sophistication has room to pay for itself underneath that.**

This is a fifty-year-old theorem wearing modern clothes. Hecht–Ullman and
Kam–Ullman showed that round-robin iterative dataflow on a *reducible* flow
graph converges in d(G)+2 passes, where d(G) — the loop-connectedness — is the
maximum number of back edges on any acyclic path. Knuth's corpus study showed
d(G) is tiny in real programs. Measuring the Go toolchain in 2026 reproduced
both results exactly:

| what the corpus (std+cmd, 160,654 function-compiles) actually looks like | share |
|---|---|
| single basic block — no analysis at all | 41.6% |
| acyclic — one postorder pass, provably sufficient | 40.2% |
| loops of depth ≤ 2 — the theorem's d+2 is ~4 cheap sweeps | 17.9% |
| deeper nests (max observed depth: **5**) | 0.36% |
| irreducible functions | **one** (`(*decompressor).huffmanBlock`) |

Every SCC the SCC-based path ever processed was reducible. The single
irreducible function in the corpus routes to the iterative fallback and
converges in two sweeps.

## What this did to the algorithm question

The SCC/Bourdoncle machinery was built for abstract interpretation — infinite
lattices, widening, iteration counts that genuinely explode, where localizing
iteration matters. Liveness is the opposite problem: finite, distributive,
convergence bounded by a graph property real code keeps small. **The machinery
answers a question liveness never asks.**

Its one genuine win is real and fully attributed: −44% on a depth-10 loop nest
(gate-qualified, the effect vanishes when the SCC path is disabled and
reappears when it is enabled). But the mechanism is not SCC brilliance — it is
that the iterative path's loop-propagation step does work proportional to
Σ depth(b), a depth-quadratic term that only detonates at nesting depths that
**do not occur** in the corpus (max 5; the win needs ~10). Against that one
win: +12% on dense nests at identical topology, +6% if misdispatched broadly,
1,400 lines, and a convergence cap that was silently wrong.

End-to-end, none of it is visible: seven compiler variants — including one
that never uses SCC and one that always does — build all of std within 0.48%
of each other. The workload distribution *is* the algorithm.

## The correctness lesson: caps without proofs

The submitted code capped SCC iteration at three sweeps, with a comment
claiming two suffice for all 290k CFGs measured. Instrumented and counted:
**23.35% of real SCCs were still changing after three sweeps**, and 86.8%
needed more than two. The failure direction is unsafe — the iteration grows
monotonically from empty, so truncation under-approximates liveness, which is
the direction that reuses a live value's register.

The witness making this concrete is `runtime.forEachSpecial` — the garbage
collector's specials walker. Under the old cap it loses the same four
loop-carried values at seven blocks. The mechanism, visible sweep-by-sweep in
the Wolfram model: a value must travel backward around its cycle to its own
use point, and **every hop misaligned with the sweep order costs one sweep**.
crypto-des needs 2 sweeps (which is why the "two passes" claim was
believable), minimal-scc needs 3, forEachSpecial needs 5, FprintFunc needs 6.
No miscompile was exhibited — the register allocator did not demonstrably
exploit the missing values — but the fix (iterate to quiescence) costs nothing
measurable and removes the need to ever have this argument again.

## The measurement lesson: infrastructure lies structurally

Nothing in this project lied maliciously; five things lied structurally:

1. A build harness printed `BUILD OK` immediately after `go tool dist: FAILED`
   and "timed" a nonexistent binary at 1 ms. A conclusion ("threshold change
   breaks the bootstrap") lived in STATUS for a day on fabricated evidence.
2. A differential probe reported perfect agreement while comparing 16,121
   functions against an empty dump.
3. Ten of twelve `Irreducible_*` benchmarks never reached the code they were
   named for, and the flagship "irreducible diamond" was a DAG — crossing
   edges are not irreducibility; a cycle with two entries is.
4. A codegen comparison's entire signal for one variant sat below its own
   unmeasured nondeterminism floor (27 functions differ between two builds of
   the *same* compiler).
5. The experimenter ssh-ing into the benchmark box raised the noise floor
   13-fold, silently.

The countermeasures are codified in `vmbench/harness/`: builds guilty until
proven innocent, an A/A gate that must pass before any comparison is printed,
audited one-line arm diffs, no contact during measurement, every effect quoted
with its floor, and known-broken results pinned as *non-empty* test
expectations so they cannot pass vacuously.

## What is still open

- **The density flip.** At identical 13-block topology, SCC wins the sparse
  nest and loses the dense one by 12%. Hypothesis: Go's liveness lattice is
  (value, distance) pairs, and alternating sweep orders re-relax distances
  that a fixed order stabilizes sooner. Testable in the Wolfram model by
  propagating distances — which is also the remaining reviewer question
  (`regalloc.go:2893`).
- **No miscompile exhibited** for the capped liveness, despite the
  under-approximation being real. Unsoundness-in-principle plus a free fix
  carried the argument instead.
- **Corpus breadth**: std+cmd is one (large) corpus. The LIVEDUMP hook makes
  extraction mechanical if a counter-example corpus is ever proposed.

## The verdict, in one paragraph

The acyclic fast path stands on simplicity: it serves 82% of function-compiles
with a provably sufficient single pass, changes no generated code beyond the
measured nondeterminism floor, and costs nothing. The SCC path solves a
problem the corpus does not contain, at a price the benchmarks understate and
real compilation cannot see; its honest justification is one benchmark shape
absent from real code, and its history includes a correctness cap that was
wrong for a quarter of what it processed. Real-world CFGs behave differently
from the graphs that make clever algorithms attractive — they are shallow,
reducible, and overwhelmingly trivial, and the algorithm that respects that
distribution is a linear scan in the right order.

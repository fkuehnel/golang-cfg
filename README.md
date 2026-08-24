# Live variable analysis in the Go compiler

Research on the live-variable analysis that feeds the Go compiler's register
allocator, together with an independent reference implementation used to check
the compiler's answers, and the sorting workloads used to exercise it.

Companion to Gerrit **[CL 731660](https://go-review.googlesource.com/c/go/+/731660)**
and to `cfg_realworld.pdf` in this directory.

## Start here

**[`STATUS.md`](STATUS.md)** is the inventory: what is verified, what is only
partially true, what is reproduced as broken, and what is merely believed. Every
row records how it was checked. It is deliberately candid, including about this
work's own claims -- several things previously treated as results are marked
UNVERIFIED there, and the reasons are given.

## Layout

| path | what |
|---|---|
| `STATUS.md` | the inventory. Read this before trusting any number here |
| `cfg_realworld.pdf` | the companion write-up |
| `go-code/` | the compiler variants under study. **Stale** -- see STATUS.md; Gerrit patchset 13 is canonical |
| `wolfram/` | independent reference implementation of liveness, plus corpus and tests |
| `sortAlgorithms/` | sorting workloads, including a branchless quicksort port |
| `analysis/` | CFG structure statistics over the Go toolchain |
| `LVA/` | live-value comparison outputs |

## The reference implementation

`wolfram/` contains a from-scratch implementation of live-variable analysis in
Wolfram Language. It parses the Go compiler's own register-allocator debug dumps,
recomputes liveness independently, and compares. `FindDifference` returning an
empty association means the two agree.

This is a differential oracle: agreement is evidence the compiler is right,
disagreement is a bug in one of them. It currently compares value identities;
extending it to use-distances is the next step.

Run the tests from a licensed kernel:

```
SetEnvironment["LIVERANGE_ROOT" -> "<path to wolfram/>"];
TestReport["tests/LiveRange.wlt"]
```

Eleven tests, all passing, including a guard that fails loudly if the corpus
does not load -- without it, an empty corpus makes every oracle test pass
vacuously by comparing nothing to nothing.

## A result worth stating plainly

The three-pass liveness algorithm is **not correct in general**. Pass two walks
the postorder from the first element of the forward postorder; nothing
guarantees that block reaches every other block. On one real Go CFG in the
corpus it is an exit block, so pass two covers 1 of 8 blocks and does nothing,
and block `b9` loses two live values. On another CFG the same construction
happens to cover everything and the algorithm appears to work.

The iterative fixed-point solver matches the compiler on both.

## Why sorting is here

The sort code is the workload, not a side project. It drives SSA and
register-allocator dumps, and the `BenchmarkComputeLive` suite includes
`HeapSort`. There is also a sharper link: Go does not emit conditional moves
where C++ does, which forces a hand-written mask trick. Conditional-move
emission is decided by the SSA backend and the register allocator -- the same
subsystem this research modifies.

`sortAlgorithms/hybrid/` is a Go port of the Hoare-Lomuto hybrid branchless
quicksort from Gerben Stavenga's
[*Hoare's rebuttal and bubble sort's comeback*](https://github.com/gerben-s/quicksort-blog-post).

## Note on this repository

This tree is **generated** from a working repository by a whitelist script; it
is not edited directly. An earlier arrangement kept two hand-maintained copies,
which drifted apart without anyone noticing.

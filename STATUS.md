# Status inventory

What works, what does not, and what is merely believed. Last updated 2026-08-24.

Every row carries how it was checked. Rows marked **unverified** are not claims —
they are open questions. The purpose of this file is that no number or assertion
leaves this repo without its provenance attached.

| tag | meaning |
|---|---|
| **VERIFIED** | reproduced from a clean state, with a command or test that re-runs |
| **PARTIAL** | true under stated conditions, not in general |
| **BROKEN** | reproduced as failing |
| **UNVERIFIED** | believed, asserted, or measured under conditions that cannot support the claim |

---

## Algorithms

| item | status | evidence |
|---|---|---|
| Iterative fixed-point liveness matches the Go compiler | **VERIFIED** | `wolfram/tests/LiveRange.wlt`, clean kernel, both corpus CFGs, `FindDifference -> <||>` |
| Three-pass liveness (Wolfram prototype) on `crypto-des` | **PARTIAL** | true for that CFG only; `poBwd` happens to cover 14/14 blocks |
| Three-pass liveness (Wolfram prototype) in general | **BROKEN** | on `minimal-scc`, `b9` loses `v37` and `v41`. Pass 2 walks postorder from `First[poFwd]` = `b13`, an exit block with no successors, so it covers **1 of 8** blocks and is a no-op. Pinned in `minimal-scc/three-pass-is-wrong` |
| "Three-pass produces smaller distances, hence more converged" | **UNVERIFIED** | asserted in Gerrit PS9. On `minimal-scc` smaller means *missing values*, not converged. Competing explanation not yet excluded |
| Acyclic fast path is correct | **UNVERIFIED** | no differential check has been run against it |
| Use-distances match the previous algorithm | **UNVERIFIED** | Morsing's central request, asked three times. The oracle compares value *identities only*; distances are parsed but stripped before solving |
| SCC/Bourdoncle approach is slower for nested loops | **UNVERIFIED** | laptop measurement, noise-dominated |
| Claim at `regalloc.go:2983` that loop liveness is O(B² × V) | **BROKEN -- settled, Morsing is right** | Measured by instrumenting the fixup and compiling std+cmd, 23712 functions: 1 pass 85.15%, 2 passes 14.85%, never more. `visits/B` flat at 1.00-1.04 across every size bucket from B=2 to B=3135; a quadratic term would show as that ratio growing. Fixed in `scc-converge` |
| `propagateLoopLiveness` is Frank's code | **NO** | its body is upstream's inline loop-propagation code from `computeLive`, extracted verbatim into a function, `TODO(dmo)` included. Only the signature and a moved `computeDesired()` call differ |
| `computeLiveWithSccs` inherits the prototype's coverage defect | **NO -- settled** | `probes/scc-order-coverage`. The compiler's DFS is confined to the SCC, where strong connectivity makes the second order total: 0 failures in 5004 SCCs. The prototype, running over the whole CFG, fails on 51.6% of 20000 random graphs |
| `computeLiveWithSccs` 3-pass cap is sufficient | **NO -- settled** | `probes/scc-convergence`. Over 200000 random SCCs, 0.0795% yield a wrong result under the cap. **All 159 failures are under-approximations** -- missing live values, never extra. The iteration grows monotonically from empty, so truncation always yields a subset of the fixed point |
| Real Go CFGs reach those bad cases | **YES -- settled** | Compiling std+cmd with the iteration counter exposed: 1731 SCCs needed >=3 changing sweeps, of which **484 were still changing after the third** (352 at 4, 123 at 5, 9 at 6). Ordinary functions: `(*Checker).declStmt`, `nilcheckelim`, `(*labelScope).blockBranches`, `(*regAllocState).shuffle` |
| 3-pass cap produces a demonstrated miscompile | **NOT SHOWN** | codegen for `nilcheckelim` is byte-identical between the capped and converged toolchains despite 6 sweeps. The only difference found in an affected function (`fprintFunc`) is a jump-target offset, a layout artifact. Incomplete liveness is unsound in principle, but no wrong code has been exhibited |
| Convergence loop fixes it | **VERIFIED** | `backups/go-tree/scc-converge.patch`. Both loops iterate to no-change, matching baseline `computeLive`/`computeDesired`. `cmd/compile/internal/ssa` tests pass; `build -a std` 7.38-7.91s capped vs 7.61-7.80s converged, no measurable cost |
| `Func.sccs()` branch selection | **BROKEN** | `if ln != nil && !ln.hasIrreducible` — both branches call `w.kosaraju(po)`. Dead conditional |

## Measurement

| item | status | evidence |
|---|---|---|
| All December 2025 timing numbers | **SUPERSEDED, and one was wrong** | re-measured on a quiet VM. The acyclic claim holds directionally but is smaller than stated. The nested-loop claim that SCC is far slower and worsens with depth is **contradicted**: `Nested_10` is 35.8% *faster*, not 79% slower |
| Benchmark harness | **BUILT and run** | design recorded in the working repository, not published here |
| Benchmark results | **VERIFIED** | `vmbench/results/20260824T190007/`, 20 samples per benchmark per arm, interleaved with reversed order on alternate rounds, pinned core, GOMAXPROCS=1 |
| `regalloc_bench_test.go` is a stable instrument | **VERIFIED** | byte-identical across all three copies, md5 `beaa0068531f18dbf8b14c18cf96a5c6`, 1900 lines |
| Variant files are drop-in for one bounded region | **VERIFIED** | `regalloc_master.go` is byte-identical to baseline `regalloc.go` lines 2827-3227 |

## Tooling

| item | status | evidence |
|---|---|---|
| `LiveRange.wl` package | **VERIFIED** | 11/11 tests from a clean kernel; CodeInspector clean |
| Corpus loading guard | **VERIFIED** | `guard/corpus-actually-loaded`. Added after the first suite reported passes while loading zero data |
| `LaTeXTikZ` / `ExportToTikZ` | **BROKEN** | reported not working correctly; undiagnosed. Loads via `PacletDirectoryLoad["<a directory outside the repository>"]`, an absolute path outside the repo |

## Where the canonical code lives

| item | status | notes |
|---|---|---|
| Gerrit CL 731660 | **canonical**, open | patchset 13, 7 Jan 2026, 4 unresolved comments. Fetch with `git fetch https://go.googlesource.com/go refs/changes/60/731660/13` |
| `go-code/` in this repo | **stale** | roughly patchset 10-11. Retained because the publication references it. Treat Gerrit as authoritative |

## Open review items (Gerrit CL 731660)

Patchset 13 = `9eeaae54d1`, fetched 2026-08-24 via
`git fetch https://go.googlesource.com/go refs/changes/60/731660/13`.

**Its parent is `ad91f5d241`** -- the same base as patchset 1. The CL has never
been rebased, so the frozen baseline used for benchmarking is exactly this CL's
own base. No rebase is needed to compare them.

`ps11 -> ps13` changed almost nothing: `likelyadjust.go` reverted, and **four
lines** of `regalloc.go` (a debug string rename and one doc comment). The commit
message is byte-identical between the two.

| # | location | comment | status |
|---|---|---|---|
| 1 | `/COMMIT_MSG:35` | *"This new commit message says that the SCC based approach isn't justified, but it still exists in the patchset. Which is it?"* | **ADDRESSED by splitting.** Two stacked CLs on branches `acyclic-only` and `scc-on-top`. Together they reconstitute the fixed ps13 exactly (verified: the trees are identical). Whether CL B should be mailed at all is now a separate decision rather than a contradiction inside one change |
| 2 | `likelyadjust.go:51` | *"Why the changes in this file? None of the functionality changed here."* | **RESOLVED in ps13** -- the 99-line change was reverted. Not yet marked resolved in Gerrit |
| 3 | `regalloc.go:2893` | *"Again, I think this is is actually important. Do you have some basis for believing this is true?"* | **OPEN, now answerable.** Three passes are *not* sufficient in general -- see `probes/scc-convergence`. The failures are under-approximations, so the unsafe direction. Whether real Go CFGs reach them is still open |
| 4 | `regalloc.go:2983` | *"This is wrong. The loop liveness approach is specifically used because it is linear with the number of blocks."* | **RESOLVED.** He is right, and it is a documentation defect rather than an implementation one: the function is upstream's own code extracted verbatim. Measured linear over 23712 functions. Comment corrected in `scc-converge` |

Comment 4 is settled, and an earlier guess recorded here was wrong. It is a
**documentation defect, not a regression**: `propagateLoopLiveness` does not
replace anything, it *is* upstream's loop-propagation code, lifted verbatim out
of `computeLive` into a function. Frank attached an `O(B² × V)` note to code he
did not write, and the note is not merely loose but misleading -- that code was
chosen for being linear.

Instrumenting the distance fixup and compiling std and cmd (23712 functions):
one pass for 85.15%, two for 14.85%, never more; the largest function, B=3135,
converged in one. Block visits stayed at 1.00-1.04 times the block count in
every size bucket. Corrected on branch `scc-converge`.

Comment 3 has now been split into two questions and both were tested
(`probes/`):

- **Does `computeLiveWithSccs` inherit the prototype's coverage defect?** No.
  Its DFS is confined to the SCC, and strong connectivity makes the second
  traversal order total. Settled, 0 failures in 5004 SCCs.
- **Are three passes sufficient?** No. 0.0795% of random SCCs are wrong under
  the cap, and every failure is an under-approximation -- missing live values,
  which is the direction that miscompiles rather than the direction that merely
  costs performance.

That question is now settled too: compiling std and cmd, **484 SCCs were still
changing after the third sweep** and were being truncated. They are ordinary
functions, not pathological ones.

The fix is committed on branch `scc-converge`
(`backups/go-tree/scc-converge.patch`): both loops iterate until a sweep reports
no change, which is how the baseline `computeLive` and `computeDesired` already
work. Tests pass and compile time is unchanged within noise.

What is *not* shown is a miscompile. `nilcheckelim` generates byte-identical code
under both toolchains despite needing six sweeps, so an incomplete live set does
not always reach codegen -- the allocator may simply not have reused the
register. The argument for the fix is that incomplete liveness is unsound in
principle and the correction is free, not that a wrong-code bug has been
exhibited.

Note also that the code comment claims *"Two passes are sufficient for ALL SCCs
in our 290k-CFGs dataset"* while the loop is capped at three.

### The CL, split in two

| branch | subject | size |
|---|---|---|
| `acyclic-only` | *cmd/compile/internal/ssa: add an explicit acyclic fast path to liveness* | 2 files, 2165 insertions |
| `scc-on-top` | *cmd/compile/internal/ssa: add SCC-based liveness for complex CFGs* | 4 files, 1415 insertions |

`acyclic-only` keeps the `Change-Id` of CL 731660, so the existing review history
stays attached to the part that is justified. `scc-on-top` carries a fresh one.

The seam is clean: of the eight functions the CL adds to `regalloc.go`, only
`computeLiveWithSccs` references SCC machinery at all. `processDesiredWithOrder`
has a second caller in `computeDesired`, so it stays in the first CL;
`processBlocksWithOrder` and `allLoopsSimple` have no callers outside the SCC
path and move to the second.

Verified lossless: after applying both, the `ssa` tree is byte-identical to
`scc-converge`. Each branch builds and passes `cmd/compile/internal/ssa` tests
independently.

**CL B's justification remains the open question, and the split makes that
visible rather than resolving it.** Its own commit message says so: the SCC
decomposition is not a performance win -- benchmarks show it slower than the
dominator-based loopnest on nested loops, increasingly so with depth -- and the
case for it is robustness on CFGs the loopnest handles poorly, chiefly
irreducible ones. Whether that carries 1400 lines is a call for the author and
reviewers, not something measurement settles.

### Also still present in ps13

`Func.sccs()` retains the conditional whose branches are identical:

```go
if ln := f.cachedLoopnest; ln != nil && !ln.hasIrreducible {
    // something is broken here?
    //f.cachedSCCs = ln.buildSCCs()
    f.cachedSCCs = f.computeSCCs()
} else {
    f.cachedSCCs = f.computeSCCs()
}
```

Dead branch, commented-out code, and a `// something is broken here?` note, in a
change under active review.

### scc.go now has five distinct versions

| source | lines | md5 |
|---|---|---|
| **Gerrit ps13 (canonical)** | **467** | `74e2e1b5ce8c31764e92169293bae7f3` |
| Gerrit ps1 | 410 | `deca908a2a2353028fe485efeb0b0426` |
| local go tree working copy | 376 | `22ad2adb0adece406b2c904447ebd26c` |
| `go-code/` in this repo | 373 | `60e22b2d8f65429941f3cf8dbe42811d` |

ps13 matches none of them.

## Quicksort

A Go port of **Gerben Stavenga, "Hoare's rebuttal and bubble sort's comeback"**
(`github.com/gerben-s/quicksort-blog-post`; local clone
a local clone). Reference: `hybrid_qsort.h`,
`HoareLomutoHybridPartition` at lines 236-258.

The blog's contribution is not "branchless Lomuto" -- that is Alexandrescu's.
It is that branchless Lomuto carries a **store-to-load dependency** (it loads
from the partition index and stores to it; the addresses may alias, so the CPU
will not speculate the load past the store), turning a nominally 1-cycle loop
into ~8. `distributeForward` removes the aliasing load by writing larger
elements to a scratch buffer, but is no longer in place. The **hybrid** restores
in-place operation: use a small fixed scratch, and when it fills, the region it
vacated becomes the scratch for a backward distribution. Alternate.

### New implementation: `sortAlgorithms/hybrid/` (2026-08-24)

| item | status | evidence |
|---|---|---|
| Hoare-Lomuto hybrid, alternating forward/backward | **VERIFIED** | `hybridPartition`; a coverage test asserts the alternating loop actually executes (11 iterations on the test corpus), because an earlier version of the suite passed while exercising only the early-exit path |
| Branchless partitioning applied to large arrays | **VERIFIED, inversion fixed** | `len(a) > ScratchSize` now selects the hybrid; the old code sent exactly those arrays to branchy Hoare |
| `BubbleSort2` | **VERIFIED ported** | register-resident pair, two elements bubbled per pass, no unpredictable branches |
| Branchless fast path matches a safe reference | **VERIFIED** | differential oracle: branchy reimplementations of both distribute loops; fast and safe must agree on returned counts, array contents and scratch contents |
| Test suite can actually fail | **VERIFIED** | mutation tested, 8 injected defects, all caught. `medianOfThree` needed its own test -- a bad pivot costs performance without breaking correctness, so no correctness test can see it |
| Scratch reentrancy | **FIXED** | per-call, never package level. `TestSortConcurrent` under `-race`, 16 goroutines |
| `unsafe.Pointer` usage | **FIXED** | `unsafe.Add` on live pointers, no `uintptr` round trips. `go vet` clean |
| Allocation | **VERIFIED** | 0 allocs/op; inputs <= `SmallSortThreshold` skip the scratch entirely |
| Timings | **UNVERIFIED** (laptop, indicative only) | n=100000: 19.5 vs `slices.Sort` 51.8 ns/elem. n=1000: 8.4 vs 8.9, not separable from noise. The large-n gap is far outside noise, but the figure needs the quiet VM |
| Comparator support (`SortFunc`) | **not done** | a call in the inner loop would defeat the branchless property |
| Integration into `src/sort` | **not done** | the package is standalone; porting it into the Go tree is a further step |

### Legacy generators (superseded by the above)

| item | status | evidence |
|---|---|---|
| `gen_hybrid_sort.go` `hybridPartition` | **RENAMED** to `hoarePartition` | commit `967f4f7362` on branch `quicksort_hybrid`. It is classic branchy Hoare; the name claimed otherwise |
| `gen_sort_variants_hybrid.go` `hybridBlockPartition` | not the hybrid | block-based Hoare with branchy inner loops (`if Less { if readPos != writePos { Swap } }`). No backward distribute |
| `gen_sort_variants_hybrid.go` `hybridBubbleSort2` | **partially correct** | *Correction to an earlier entry in this file.* It does sort correctly on its own -- tested, 0/2000 random inputs left unsorted -- so the trailing "final cleanup pass for correctness" is redundant, not load-bearing. But it is branchy and swaps through memory rather than keeping the pair in registers, so it is not the reference algorithm and does not get the 2-cycle loop |
| "18% faster than pdqsort" (original claim) | **superseded** | measured against the old implementation, in which the branchless loop ran only on partitions of at most 128 elements |

## Next increments, in dependency order

1. **Distances in the oracle.** `LiveRange.wl` parses distances and discards them before solving. Propagating them through the transfer function is what converts this infrastructure into an answer for `regalloc.go:2893`.
2. **Corpus extraction** from the archived dumps, so the oracle runs over hundreds of CFGs rather than two.
3. **The quiet VM**, for timing. Independent of 1 and 2 — correctness is hardware-independent, timing is not.
4. **Cheap review cleanup**: revert `likelyadjust.go`; remove the dead conditional in `Func.sccs()`; resolve the commit-message contradiction.
5. **Quicksort**: algorithm done and tested standalone. Remaining: port into `src/sort`, add comparator support, tune the two thresholds, and measure on the VM.

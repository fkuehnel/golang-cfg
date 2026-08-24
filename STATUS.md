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
| Three-pass liveness matches the compiler on `crypto-des` | **PARTIAL** | true for that CFG only; `poBwd` happens to cover 14/14 blocks |
| Three-pass liveness in general | **BROKEN** | on `minimal-scc`, `b9` loses `v37` and `v41`. Pass 2 walks postorder from `First[poFwd]` = `b13`, an exit block with no successors, so it covers **1 of 8** blocks and is a no-op. Pinned in `minimal-scc/three-pass-is-wrong` |
| "Three-pass produces smaller distances, hence more converged" | **UNVERIFIED** | asserted in Gerrit PS9. On `minimal-scc` smaller means *missing values*, not converged. Competing explanation not yet excluded |
| Acyclic fast path is correct | **UNVERIFIED** | no differential check has been run against it |
| Use-distances match the previous algorithm | **UNVERIFIED** | Morsing's central request, asked three times. The oracle compares value *identities only*; distances are parsed but stripped before solving |
| SCC/Bourdoncle approach is slower for nested loops | **UNVERIFIED** | laptop measurement, noise-dominated |
| Claim at `regalloc.go:2983` about loop-liveness complexity | **BROKEN** | Morsing: *"This is wrong. The loop liveness approach is specifically used because it is linear with the number of blocks."* Not yet re-examined |
| `Func.sccs()` branch selection | **BROKEN** | `if ln != nil && !ln.hasIrreducible` — both branches call `w.kosaraju(po)`. Dead conditional |

## Measurement

| item | status | evidence |
|---|---|---|
| All December 2025 timing numbers | **UNVERIFIED** | laptop, batched arms, effect size 2-3% below the noise floor. Not reproducible; hardware no longer available for the purpose |
| Benchmark harness design | designed, **not built** | design recorded in the working repository, not published here |
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

## Open review items (Gerrit CL 731660, patchset 11, all Morsing)

| location | comment | needs |
|---|---|---|
| `/COMMIT_MSG:35` | *"This new commit message says that the SCC based approach isn't justified, but it still exists in the patchset. Which is it?"* | scope decision: split the SCC work out, or justify it |
| `likelyadjust.go:51` | *"Why the changes in this file? None of the functionality changed here."* | revert the file |
| `regalloc.go:2893` | *"Again, I think this is is actually important. Do you have some basis for believing this is true?"* | the distances oracle |
| `regalloc.go:2983` | *"This is wrong. The loop liveness approach is specifically used because it is linear with the number of blocks."* | correct it, or defend it |

Morsing's standing request from PS9: *"Have you verified that the use-distances are appropriate in this CL? ... See CL 694696 for an example of this verification."* CL 694696 is **"cmd/compile: debug version of liveness"**, 495 insertions, created Aug 2025, **abandoned** Nov 2025 — an instrumented compiler built to make liveness diffable.

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

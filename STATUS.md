# Status inventory

What works, what does not, and what is merely believed. Last updated 2026-08-25.

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
| Acyclic fast path is correct | **PARTIAL** | `probes/dispatch-census/divergence_test.go`: all 12 `Irreducible_*` CFGs agree with an iterate-to-fixed-point reference on value IDs *and* distances -- including the 10 that are cyclic yet dispatched to the acyclic path. No divergence exhibited; not a general proof |
| `len(Loops)==0` implies the CFG is acyclic | **NO -- settled** | `loopnestfor` skips irreducible SCCs (`if !scc.IsReducible() { sawIrred = true; continue }`), so they never become Loops. `Irr_Simple`: 4 blocks, 0 loops, `HasIrreducible=true`, cyclic, dispatched to `computeLiveAcyclic`. The comment "No loops = no cycles" is false. Upstream's own field note already warns: *"if accurate loops are needed, must punt at function level"* |
| The `Irreducible_*` benchmarks exercise the SCC path | **NO -- settled** | `probes/dispatch-census`. 10 of 12 take the ACYCLIC path, `Irr_WithReducible` takes ITER, **none reach `computeLiveWithSccs`**. CL B's stated justification -- robustness on CFGs the loopnest handles poorly -- has no benchmark evidence behind it |
| `Irreducible_Diamond` is irreducible | **WAS NO -- FIXED 2026-08-25** | it had zero back edges and was a plain DAG. The "cross edges" L1->R2 / R1->L2 run sideways between forward levels and never close a cycle; crossing edges are not irreducibility, a cycle with two entry points is. Fixed by adding back edges L2->L1 and R2->R1, making {L1,R1,L2,R2} one SCC entered at two distinct blocks. All six VM arms updated, bench md5 `a279eda7` -> `6ecf446a`. Still routes to the acyclic path, so the census is unchanged |
| The `allLoopsSimple` threshold is tuned | **NO -- swept 2026-08-25, benchmark level only** | The CL ships `allLoopsSimple(2)`; nobody had varied it. It changes the path for exactly 5 benchmarks, so the other 19 are a free control. vs base: T=2 regresses `Nested_3` +6.5%, `NestedDense_3x5` +12.8%, `NestedDense_5x20` +13.2%; T=3 clears the first two; **T=5 clears all three and regresses nothing**, geomean -2.98% vs T=2's -1.58%. `Nested_10`'s -43% is invariant throughout. **CAVEAT:** this was swept in the standalone `ssacompile` benchmark package, which does not build the compiler -- see the `SCC path survives a lower dispatch threshold` row, where `allLoopsSimple(3)` failed to bootstrap. The two must be reconciled before any threshold change goes into the CL |
| Real-corpus dispatch census exists | **YES -- 2026-08-25** | LIVESTATS-instrumented compiler over std+cmd, 160654 function-compiles (`vmbench/results/20260825T211912-census/`, analyzer `vmbench/harness/livestats.py`): single-block 41.59%, acyclic 40.20%, iter@T=2 17.85%, **scc@T=2 0.36%** (576). At T=3: 0.06% (101). At T=4: five functions. **At T>=5: zero -- no function in std+cmd has loop depth >5** (depth histogram: 1:22913, 2:5767, 3:475, 4:96, 5:5). So T=5 on real code means the SCC path never executes, and the microbench T=5 'win' is a statement about benchmark CFGs only |
| The SCC path ever sees an irreducible CFG in practice | **NO -- settled on std+cmd** | All **2051 of 2051** non-trivial SCCs processed by the SCC path are reducible (`red=false` count: 0). The corpus contains exactly **one** irreducible function -- `(*decompressor).huffmanBlock`, compiled twice -- and it has maxdepth=1, so it routes to the ITERATIVE path, converging in 2 sweeps. The SCC path's stated justification (robustness on irreducible CFGs) is exercised by nothing, anywhere in std+cmd |
| 3-pass cap on REAL code (clean protocol) | **WRONG for 23.35% of SCCs -- settled** | Converge-and-count instrumentation, same build: of 2051 non-trivial SCCs, changing-sweep histogram 1:76, 2:194, 3:1302, 4:347, 5:125, 6:7. **479 (23.35%) still changing after the third sweep** -- the old cap under-approximated their liveness (confirms the earlier 484 under a cleaner protocol). **1781 (86.84%) need more than two changing sweeps**, so the code comment "Two passes are sufficient for ALL SCCs in our 290k-CFGs dataset" is false for 87% of the SCCs the path actually processes. Random-graph probes said 0.08% fail; real reducible nests fail at 23% -- real code is ~300x worse than random for this cap. Worst offenders all reducible, lsweeps=6: `findDeclsAndUnresolved`, `(*labelScope).blockBranches`, `FprintFunc`, `checkBranches` |
| A real >3-sweep witness exists in inspectable form | **YES -- 2026-08-26, cross-validated** | `LIVEDUMP` hook emits corpus-format dumps for any named function; `wolfram/corpus/forEachSpecial` (runtime mheap specials walker, 23 blocks, 19-block SCC) and `wolfram/corpus/FprintFunc` (78 blocks) extracted. The Wolfram `SccSweepSolution` model independently reproduces the in-compiler sweep counts (5 and {1,6}) and matches the compiler's converged live sets exactly on both. Under the old 3-cap: forEachSpecial loses the SAME four loop-carried values (`v3,v13,v66,v68`) at seven blocks -- four sweeps still miss b35 -- and FprintFunc is wrong at **37 of 78 blocks**. Pinned in `tests/LiveRange.wlt` (24/24 passing, fresh kernel). The sweep-cost law, visible in `SccSweepTrace`: a value must travel backward around its cycle to its own use point, and every order-misaligned hop costs one sweep |
| Codegen is identical across dispatch variants | **NO -- but structured and tiny** | Per-function asm digests of std (33948 functions, `codegen.log`): aclonly differs from base in 27 functions (0.08%), sccT2 in 97 (0.29%), sccT3 in 37, sccT5 in 24. The pattern tracks dispatch: `crypto/des.initFeistelBox` (depth 4) differs at T=2/T=3 but not T=5; `compress/flate.*encode` only at T=2. Mechanism: path choice changes use-distances, distances steer allocation. 2026-08-26 refinement: base digested twice differs in **27 functions -- the comparison's nondeterminism floor**. aclonly's 27 diffs sit AT that floor (17/27 in the observed draw) -- no evidence of real change. sccT2 keeps **79 functions of real signal** above the floor, all loopy (flate encoders, bzip2, x509 walkers, `initFeistelBox`). The one inspected diff, `initFeistelBox` base-vs-sccT2 path-normalized: different register walk, same spill count (41), **6 bytes smaller** under SCC (236->230). Different, not worse, in the case examined |
| Nested_10's -43% belongs to the SCC path | **YES -- settled, gate-qualified** | 7-arm sweep `vmbench/results/20260826T000827-final/`, A/A gate passed (0/25 significant, floor -0.65%). `iterAll` (SCC path disabled, all loopy functions on the CL's iterative path) is FLAT vs base: `Nested_10` p=0.201, geomean +0.41% ~ floor. iterAll->sccT2: `Nested_10` **-44.12%**. The win is 100% SCC's. So are the losses: `NestedDense_3x5` +12.22%, `NestedDense_5x20` +12.16%. The CL's iterative rewrite itself is performance-neutral vs upstream |
| The CL is visible in real compile time | **NO -- settled at 1% resolution** | Timed `go build -a std` on isolated cores, 8 recorded rounds per arm, interleaved: all seven arms (base, sccT2/T3/T5, iterAll, sccAll, aclonly) mean 41.8-42.0s, spread 0.48%, per-arm sd ~1.5%. Acyclic fast path, SCC machinery, thresholds: none of it resolves in wall-clock compile time. Effects this small are 'not resolved', not 'confirmed small' |
| Forcing SCC everywhere is safe | **NO** | `sccAll` arm (dispatch always SCC for loopy functions): microbench geomean **+6.19%** vs base -- shallow loops pay the SCC overhead with nothing back. The dispatch threshold exists for a reason; the question is only whether the path behind it earns its 1400 lines |
| Use-distances match the previous algorithm | **UNVERIFIED** | Morsing's central request, asked three times. The oracle compares value *identities only*; distances are parsed but stripped before solving |
| SCC/Bourdoncle approach is slower for nested loops | **SETTLED, and the question was wrong** | It is faster on sparse nests and slower on dense ones *at identical CFG topology* -- `Nested_5` -3.90% against `NestedDense_5x20` +12.64%, same 13 blocks, same single SCC. The predictor is value density, not loop structure. See the 2026-08-25 benchmark section |
| Claim at `regalloc.go:2983` that loop liveness is O(B² × V) | **BROKEN -- settled, Morsing is right** | Measured by instrumenting the fixup and compiling std+cmd, 23712 functions: 1 pass 85.15%, 2 passes 14.85%, never more. `visits/B` flat at 1.00-1.04 across every size bucket from B=2 to B=3135; a quadratic term would show as that ratio growing. Fixed in `scc-converge` |
| `propagateLoopLiveness` is Frank's code | **NO** | its body is upstream's inline loop-propagation code from `computeLive`, extracted verbatim into a function, `TODO(dmo)` included. Only the signature and a moved `computeDesired()` call differ |
| `computeLiveWithSccs` inherits the prototype's coverage defect | **NO -- settled** | `probes/scc-order-coverage`. The compiler's DFS is confined to the SCC, where strong connectivity makes the second order total: 0 failures in 5004 SCCs. The prototype, running over the whole CFG, fails on 51.6% of 20000 random graphs |
| `computeLiveWithSccs` 3-pass cap is sufficient | **NO -- settled** | `probes/scc-convergence`. Over 200000 random SCCs, 0.0795% yield a wrong result under the cap. **All 159 failures are under-approximations** -- missing live values, never extra. The iteration grows monotonically from empty, so truncation always yields a subset of the fixed point |
| Real Go CFGs reach those bad cases | **YES -- settled** | Compiling std+cmd with the iteration counter exposed: 1731 SCCs needed >=3 changing sweeps, of which **484 were still changing after the third** (352 at 4, 123 at 5, 9 at 6). Ordinary functions: `(*Checker).declStmt`, `nilcheckelim`, `(*labelScope).blockBranches`, `(*regAllocState).shuffle` |
| 3-pass cap produces a demonstrated miscompile | **NOT SHOWN** | codegen for `nilcheckelim` is byte-identical between the capped and converged toolchains despite 6 sweeps. The only difference found in an affected function (`fprintFunc`) is a jump-target offset, a layout artifact. Incomplete liveness is unsound in principle, but no wrong code has been exhibited |
| Convergence loop fixes it | **VERIFIED** | `backups/go-tree/scc-converge.patch`. Both loops iterate to no-change, matching baseline `computeLive`/`computeDesired`. `cmd/compile/internal/ssa` tests pass; `build -a std` 7.38-7.91s capped vs 7.61-7.80s converged, no measurable cost |
| SCC path survives a lower dispatch threshold | **YES -- RESOLVED 2026-08-25** | The old record was fabricated by the old harness: `build128.log` prints `BUILD OK` directly after `go tool dist: FAILED`, and `thrsweep.log` "timed" a nonexistent binary at `std build(ms): 1 1 1`. Rebuilt under paranoid detection (delete `bin/go` first; require make.bash rc=0 + binary recreated + runs + builds `os` + builds std): **all five arms bootstrap** -- base, acyclic-only, sccT2/T3/T5 -- each passing the exact step (`go build os`) where the old arms panicked. `~/matrix/SUMMARY.txt` |
| `Func.sccs()` branch selection | **BROKEN** | `if ln != nil && !ln.hasIrreducible` — both branches call `w.kosaraju(po)`. Dead conditional |

## Measurement

| item | status | evidence |
|---|---|---|
| All December 2025 timing numbers | **SUPERSEDED, and one was wrong** | re-measured on a quiet VM. The acyclic claim holds directionally but is smaller than stated. The nested-loop claim that SCC is far slower and worsens with depth is **contradicted**: `Nested_10` is 35.8% *faster*, not 79% slower |
| Benchmark harness | **BUILT and run** | design recorded in the working repository, not published here |
| Benchmark results | **VERIFIED** | three runs: `20260824T190007` (base `ad91f5d241`, history only), `20260824T215114-go127`, `20260824T233746-go128`. 20 samples per benchmark per arm, interleaved with reversed order on alternate rounds, pinned core, GOMAXPROCS=1 |
| Suite resolution below ~2% | **NOT ESTABLISHED** | the 12 irreducible benchmarks run identical code in every arm yet move -1.48% (go127) against +2.13% (go128) as a block. No A/A arm -- one commit built twice -- has ever been run, so the build-drift floor is bounded only by that accident |
| `regalloc_bench_test.go` is a stable instrument | **VERIFIED within a run, not across runs** | md5 `beaa0068531f18dbf8b14c18cf96a5c6` across all arms of `20260824T190007` and of the go127 run; `a279eda76cd70b056988c0b6170a65e8` across all arms of the go128 run. The two differ because upstream split the package: `computeLive` now lives in `cmd/compile/internal/ssacompile`. Same 25 benchmark names in both |
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

*(2026-08-25 evening: items 3 and 4 of the previous list are done -- `fuse-128`
refuted its premise, and every sweep now carries a built-in A/A gate via
`vmbench/harness/`. The threshold question resolved differently than expected:
no real function exceeds depth 5, so tuning T is moot; see the census rows.)*

1. **The CL-level decision, now fully data-backed.** CL A (acyclic-only) serves
   81.8% of function-compiles, is codegen-near-invariant (27 functions), and is
   performance-neutral in real compile time -- it stands on simplicity, not
   speed. CL B's remaining justification is exactly one thing: the deep-nest
   microbenchmark case (`Nested_10` -44%, 100% attributable to the SCC path),
   against +12% on dense nests, +6% if ever misdispatched broadly, 1400 lines,
   a 23% wrong-under-cap history requiring the convergence fix, and zero
   irreducible CFGs to serve in practice. Writing this trade-off into the CL
   thread honestly is the next mail.
2. **Distances in the oracle.** Unchanged: `LiveRange.wl` discards distances
   before solving; propagating them answers `regalloc.go:2893`.
3. **Corpus extraction at scale.** The LIVEDUMP hook in the instrumented
   compiler now emits corpus-format dumps for any named function; the
   `forEachSpecial` / `FprintFunc` witnesses are the template. Extending to
   hundreds of CFGs is mechanical.
4. **Cheap review cleanup**: revert `likelyadjust.go`; remove the dead
   conditional in `Func.sccs()`; resolve the commit-message contradiction;
   fix the `Irreducible_Diamond` doc/CFG upstream (done in all local arms).
5. **Quicksort**: unchanged -- port into `src/sort`, comparator support,
   threshold tuning, VM measurement.

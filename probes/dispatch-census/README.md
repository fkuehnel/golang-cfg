# dispatch-census

Unlike the other probes these are not `go run .` programs: they are drop-in Go
tests for `src/cmd/compile/internal/ssacompile` in a built arm, because they
need the real `regAllocState` and the benchmark CFG builders. Copy them into
that directory and run with the arm's own toolchain:

    cd $ARM/src/cmd/compile/internal/ssacompile
    GOROOT=$ARM $ARM/bin/go test -run TestDispatchCensus -v ./

## `cyclicity_test.go`

**Question.** The `Irreducible_*` benchmarks are named for a property nothing
checks. Are they actually irreducible?

**Answer.** Eleven of twelve are. `Irr_Diamond` was **not** -- it had zero back
edges and was a plain DAG. Its doc comment claimed "cross edges create
irreducibility", but L1->R2 and R1->L2 run sideways between forward levels and
never close a cycle. Crossing edges are not irreducibility; a cycle with two
entry points is. Fixed 2026-08-25 by adding back edges L2->L1 and R2->R1, which
makes {L1,R1,L2,R2} one SCC entered at two distinct blocks.

## `dispatch_census_test.go`

**Question.** Which `computeLive` path does each benchmark CFG actually take,
and at which `allLoopsSimple` threshold does that change?

**Answer.** `loopnestfor` skips irreducible SCCs (`if !scc.IsReducible() {
sawIrred = true; continue }`), so they never become Loops. The dispatch tests
`len(Loops) == 0` under the comment "No loops = no cycles", which is false.
**Ten of twelve `Irreducible_*` CFGs take the single-pass acyclic path and none
reach `computeLiveWithSccs`** -- so the suite carries no evidence for the SCC
CL's stated justification.

The threshold changes the path for exactly five benchmarks: `Nested_3` and
`NestedDense_3x5` (depth 3) flip at T=3; `Nested_5`, `NestedDense_5x3` and
`NestedDense_5x20` (depth 5) flip at T=5. `Nested_10` (depth 10) stays on the
SCC path throughout. Everything else is invariant, which makes the other 19
benchmarks a free control on any threshold sweep.

## `divergence_test.go`

**Question.** The acyclic fast path runs one postorder pass. Ten irreducible --
therefore cyclic -- CFGs are dispatched to it. Does that produce wrong liveness?

**Answer: no, not on these inputs.** All twelve agree with an
iterate-to-fixed-point reference, comparing value IDs *and* distances, including
the corrected diamond. The dispatch rests on a false premise but no divergence
has been exhibited. This is the probe to re-run against any new irreducible
shape.

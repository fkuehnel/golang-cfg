package ssacompile

import (
	"fmt"
	"strings"
	"testing"

	"cmd/compile/internal/ssa"
)

type censusCase struct {
	name  string
	build func(tb testing.TB) *ssa.Func
}

func censusCases() []censusCase {
	return []censusCase{
		{"Acyclic_500", func(tb testing.TB) *ssa.Func { return buildLinearChain(tb, 500) }},
		{"AcyclicDense_100x5", func(tb testing.TB) *ssa.Func { return buildLinearChainDense(tb, 100, 5) }},
		{"AcyclicDense_200x20", func(tb testing.TB) *ssa.Func { return buildLinearChainDense(tb, 200, 20) }},
		{"Loop_10", func(tb testing.TB) *ssa.Func { return buildSimpleLoop(tb, 8) }},
		{"Loop_100", func(tb testing.TB) *ssa.Func { return buildSimpleLoop(tb, 98) }},
		{"Nested_3", func(tb testing.TB) *ssa.Func { return buildNestedLoops(tb, 3) }},
		{"Nested_5", func(tb testing.TB) *ssa.Func { return buildNestedLoops(tb, 5) }},
		{"Nested_10", func(tb testing.TB) *ssa.Func { return buildNestedLoops(tb, 10) }},
		{"NestedDense_3x5", func(tb testing.TB) *ssa.Func { return buildNestedLoopsWithWork(tb, 3, 5) }},
		{"NestedDense_5x3", func(tb testing.TB) *ssa.Func { return buildNestedLoopsWithWork(tb, 5, 3) }},
		{"NestedDense_5x20", func(tb testing.TB) *ssa.Func { return buildNestedLoopsWithWork(tb, 5, 20) }},
		{"Irr_Simple", func(tb testing.TB) *ssa.Func { return buildIrreducibleSimple(tb) }},
		{"Irr_Diamond", func(tb testing.TB) *ssa.Func { return buildIrreducibleDiamond(tb) }},
		{"Irr_Loop5", func(tb testing.TB) *ssa.Func { return buildIrreducibleLoop(tb, 5) }},
		{"Irr_Loop10", func(tb testing.TB) *ssa.Func { return buildIrreducibleLoop(tb, 10) }},
		{"Irr_Loop20", func(tb testing.TB) *ssa.Func { return buildIrreducibleLoop(tb, 20) }},
		{"Irr_MultiEntry5", func(tb testing.TB) *ssa.Func { return buildIrreducibleMultiEntry(tb, 5) }},
		{"Irr_MultiEntry10", func(tb testing.TB) *ssa.Func { return buildIrreducibleMultiEntry(tb, 10) }},
		{"Irr_Nested5", func(tb testing.TB) *ssa.Func { return buildIrreducibleNested(tb, 5) }},
		{"Irr_Nested10", func(tb testing.TB) *ssa.Func { return buildIrreducibleNested(tb, 10) }},
		{"Irr_WithReducible", func(tb testing.TB) *ssa.Func { return buildIrreducibleWithReducible(tb, 5) }},
		{"Irr_Chain3", func(tb testing.TB) *ssa.Func { return buildIrreducibleChain(tb, 3) }},
		{"Irr_Chain10", func(tb testing.TB) *ssa.Func { return buildIrreducibleChain(tb, 10) }},
	}
}

func pathFor(ln *ssa.LoopNest, thr int16) string {
	if len(ln.Loops) == 0 {
		return "ACYCLIC"
	}
	if allLoopsSimple(ln, thr) {
		return "ITER"
	}
	return "SCC"
}

// runVariant computes liveness with a chosen algorithm and returns the dump.
func runVariant(f *ssa.Func, which string) (res string, err any) {
	defer func() {
		if r := recover(); r != nil {
			err = r
		}
	}()
	s := &regAllocState{}
	s.init(f)
	if which == "dispatched" {
		s.computeLive()
		return dumpLiveSets(f, s.live), nil
	}
	po := f.Postorder()
	s.live = make([][]liveInfo, f.NumBlocks())
	s.desired = make([]desiredState, f.NumBlocks())
	s.loopnest = f.Loopnest()
	s.loopnest.ComputeUnavoidableCalls()
	lv := f.NewSparseMapPos(f.NumValues())
	defer f.RetSparseMapPos(lv)
	tt := f.NewSparseMapPos(f.NumValues())
	defer f.RetSparseMapPos(tt)
	switch which {
	case "iterative":
		s.computeLiveIterative(po, lv, tt)
	case "scc":
		s.computeLiveWithSccs(po, lv, tt)
	}
	return dumpLiveSets(f, s.live), nil
}

// TestDispatchCensus reports, for every benchmark CFG, which computeLive path
// it takes at each candidate allLoopsSimple threshold, and whether the
// dispatched result matches an iterate-to-fixed-point reference.
func TestDispatchCensus(t *testing.T) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "\n%-20s %6s %6s %6s %6s  %-8s %-8s %-8s %-8s  %-9s %-9s\n",
		"CFG", "blocks", "loops", "irred", "maxD",
		"T=1", "T=2", "T=3", "T=4", "vs-iter", "vs-scc")
	fmt.Fprintf(&sb, "%s\n", strings.Repeat("-", 118))

	for _, c := range censusCases() {
		f := c.build(t)
		ln := f.Loopnest()

		maxd := 0
		if len(ln.Loops) > 0 {
			for d := int16(1); d <= 32; d++ {
				if allLoopsSimple(ln, d) {
					maxd = int(d)
					break
				}
			}
		}

		got, e0 := runVariant(f, "dispatched")
		iter, e1 := runVariant(f, "iterative")
		scc, e2 := runVariant(f, "scc")

		cmp := func(a string, ea any, b string, eb any) string {
			if ea != nil || eb != nil {
				return "PANIC"
			}
			if a == b {
				return "agree"
			}
			return "DIVERGE"
		}

		fmt.Fprintf(&sb, "%-20s %6d %6d %6v %6d  %-8s %-8s %-8s %-8s  %-9s %-9s\n",
			c.name, len(f.Blocks), len(ln.Loops), ln.HasIrreducible, maxd,
			pathFor(ln, 1), pathFor(ln, 2), pathFor(ln, 3), pathFor(ln, 4),
			cmp(got, e0, iter, e1), cmp(got, e0, scc, e2))
	}
	t.Log(sb.String())
}

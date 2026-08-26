package ssacompile

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"cmd/compile/internal/ssa"
)

func dumpLive(f *ssa.Func, live [][]liveInfo) string {
	var sb strings.Builder
	for _, b := range f.Blocks {
		var ids []string
		for _, li := range live[b.ID] {
			ids = append(ids, fmt.Sprintf("v%d/%d", li.ID, li.dist))
		}
		sort.Strings(ids)
		fmt.Fprintf(&sb, "b%d: %s\n", b.ID, strings.Join(ids, " "))
	}
	return sb.String()
}

// TestIrreducibleDivergence compares what computeLive actually produces on each
// irreducible CFG against an iterate-to-fixed-point reference. Every one of
// these has len(Loops)==0 -- because irreducible SCCs never become Loops -- so
// computeLive dispatches them to the single-pass acyclic path.
func TestIrreducibleDivergence(t *testing.T) {
	cases := []struct {
		name  string
		build func(tb testing.TB) *ssa.Func
	}{
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

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := c.build(t)

			s1 := &regAllocState{}
			s1.init(f)
			s1.computeLive()
			got := dumpLive(f, s1.live)

			s2 := &regAllocState{}
			s2.init(f)
			po := f.Postorder()
			s2.live = make([][]liveInfo, f.NumBlocks())
			s2.desired = make([]desiredState, f.NumBlocks())
			s2.loopnest = f.Loopnest()
			s2.loopnest.ComputeUnavoidableCalls()
			lv := f.NewSparseMapPos(f.NumValues())
			defer f.RetSparseMapPos(lv)
			tt := f.NewSparseMapPos(f.NumValues())
			defer f.RetSparseMapPos(tt)
			s2.computeLiveIterative(po, lv, tt)
			want := dumpLive(f, s2.live)

			if got != want {
				t.Errorf("DIVERGENCE\n--- computeLive() as dispatched ---\n%s--- iterate to fixed point ---\n%s", got, want)
			}
		})
	}
}

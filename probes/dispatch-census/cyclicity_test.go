package ssacompile

import (
	"fmt"
	"strings"
	"testing"

	"cmd/compile/internal/ssa"
)

// backEdges finds edges b->s where s is grey on the DFS stack, i.e. real cycles.
func backEdges(f *ssa.Func) []string {
	const white, grey, black = 0, 1, 2
	color := make([]int8, f.NumBlocks())
	var out []string
	var dfs func(b *ssa.Block)
	dfs = func(b *ssa.Block) {
		color[b.ID] = grey
		for _, e := range b.Succs {
			s := e.Block()
			switch color[s.ID] {
			case white:
				dfs(s)
			case grey:
				out = append(out, fmt.Sprintf("b%d->b%d", b.ID, s.ID))
			}
		}
		color[b.ID] = black
	}
	dfs(f.Entry)
	return out
}

func TestCyclicity(t *testing.T) {
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

	var sb strings.Builder
	fmt.Fprintf(&sb, "\n%-20s %7s %7s %8s  %s\n", "CFG", "blocks", "loops", "irred", "back edges (real cycles)")
	fmt.Fprintf(&sb, "%s\n", strings.Repeat("-", 92))
	for _, c := range cases {
		f := c.build(t)
		ln := f.Loopnest()
		be := backEdges(f)
		desc := strings.Join(be, " ")
		if len(be) == 0 {
			desc = "NONE -- graph is ACYCLIC"
		}
		fmt.Fprintf(&sb, "%-20s %7d %7d %8v  %s\n",
			c.name, len(f.Blocks), len(ln.Loops), ln.HasIrreducible, desc)
	}
	t.Log(sb.String())
}

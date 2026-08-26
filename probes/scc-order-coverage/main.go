package main

import (
	"fmt"
	"math/rand"
)

type Block struct {
	ID    int
	Succs []*Block
}

// ---- Tarjan SCC (independent of the code under test) -----------------------
func sccs(blocks []*Block) [][]*Block {
	idx := map[int]int{}
	low := map[int]int{}
	on := map[int]bool{}
	var stack []*Block
	var out [][]*Block
	next := 0
	var strong func(v *Block)
	strong = func(v *Block) {
		idx[v.ID], low[v.ID] = next, next
		next++
		stack = append(stack, v)
		on[v.ID] = true
		for _, w := range v.Succs {
			if _, ok := idx[w.ID]; !ok {
				strong(w)
				if low[w.ID] < low[v.ID] {
					low[v.ID] = low[w.ID]
				}
			} else if on[w.ID] && idx[w.ID] < low[v.ID] {
				low[v.ID] = idx[w.ID]
			}
		}
		if low[v.ID] == idx[v.ID] {
			var comp []*Block
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				on[w.ID] = false
				comp = append(comp, w)
				if w == v {
					break
				}
			}
			out = append(out, comp)
		}
	}
	for _, b := range blocks {
		if _, ok := idx[b.ID]; !ok {
			strong(b)
		}
	}
	return out
}

// ---- EXACT transcription of ps13 sccAlternatingOrdersDFS -------------------
type blockAndIndex struct {
	b     *Block
	index int
}

func sccAlternatingOrdersDFS(blks []*Block) (entryward, exitward []*Block) {
	n := len(blks)
	switch n {
	case 0:
		return
	case 1:
		entryward, exitward = blks, blks
		return
	case 2:
		entryward = []*Block{blks[1], blks[0]}
		exitward = blks
		return
	}
	inSCC := make(map[int]bool, n)
	for _, b := range blks {
		inSCC[b.ID] = true
	}
	dfsFrom := func(start *Block) []*Block {
		seen := make(map[int]bool, n)
		order := make([]*Block, 0, n)
		stack := make([]blockAndIndex, 0, n)
		seen[start.ID] = true
		stack = append(stack, blockAndIndex{b: start})
		for len(stack) > 0 {
			top := len(stack) - 1
			x := &stack[top]
			if x.index < len(x.b.Succs) {
				succ := x.b.Succs[x.index]
				x.index++
				if inSCC[succ.ID] && !seen[succ.ID] {
					seen[succ.ID] = true
					stack = append(stack, blockAndIndex{b: succ})
				}
				continue
			}
			stack = stack[:top]
			order = append(order, x.b)
		}
		return order
	}
	entry := blks[0]
	entryward = dfsFrom(entry)
	exitward = dfsFrom(entryward[0])
	return
}

// ---- the PROTOTYPE construction, over the whole graph (no SCC confinement) --
func protoOrders(all []*Block) (fwd, bwd []*Block) {
	dfs := func(start *Block) []*Block {
		seen := map[int]bool{}
		var order []*Block
		var visit func(b *Block)
		visit = func(b *Block) {
			seen[b.ID] = true
			for _, s := range b.Succs {
				if !seen[s.ID] {
					visit(s)
				}
			}
			order = append(order, b)
		}
		visit(start)
		return order
	}
	fwd = dfs(all[0])
	bwd = dfs(fwd[0])
	return
}

func randomCFG(r *rand.Rand, n int) []*Block {
	bs := make([]*Block, n)
	for i := range bs {
		bs[i] = &Block{ID: i}
	}
	for i := 0; i < n; i++ {
		k := r.Intn(3)
		for j := 0; j < k; j++ {
			t := r.Intn(n)
			bs[i].Succs = append(bs[i].Succs, bs[t])
		}
	}
	return bs
}

func main() {
	r := rand.New(rand.NewSource(42))
	trials := 20000

	sccChecked, sccIncomplete := 0, 0
	protoChecked, protoIncomplete := 0, 0

	for t := 0; t < trials; t++ {
		n := 3 + r.Intn(14)
		g := randomCFG(r, n)

		// (A) the compiler's construction, per SCC
		for _, comp := range sccs(g) {
			if len(comp) < 3 {
				continue // n<3 is special-cased and trivially total
			}
			ew, xw := sccAlternatingOrdersDFS(comp)
			sccChecked++
			if len(ew) != len(comp) || len(xw) != len(comp) {
				sccIncomplete++
				if sccIncomplete == 1 {
					fmt.Printf("  SCC counterexample: |scc|=%d entryward=%d exitward=%d\n",
						len(comp), len(ew), len(xw))
				}
			}
		}

		// (B) the prototype's construction, over the whole graph
		fwd, bwd := protoOrders(g)
		protoChecked++
		if len(bwd) != len(fwd) {
			protoIncomplete++
		}
	}

	fmt.Printf("\ncompiler sccAlternatingOrdersDFS (confined to the SCC):\n")
	fmt.Printf("  SCCs checked (|scc|>=3): %d\n", sccChecked)
	fmt.Printf("  orders failing to cover the SCC: %d\n", sccIncomplete)
	fmt.Printf("\nprototype construction (whole CFG, no SCC confinement):\n")
	fmt.Printf("  graphs checked: %d\n", protoChecked)
	fmt.Printf("  second order failing to cover the first: %d (%.1f%%)\n",
		protoIncomplete, 100*float64(protoIncomplete)/float64(protoChecked))
}

// Does the compiler's 2-order alternating scheme reach the liveness fixed point
// within the 3 passes computeLiveWithSccs allows?
//
// Simulates backward liveness on strongly connected components:
//
//	live_in[b]  = use[b] | (live_out[b] \ def[b])
//	live_out[b] = union of live_in[succ]
//
// processed block-by-block in a given order (Gauss-Seidel, as the compiler does),
// alternating entryward/exitward exactly as sccAlternatingOrdersDFS produces them.
package main

import (
	"fmt"
	"math/rand"
)

type Block struct {
	ID    int
	Succs []*Block
	def   uint64
	use   uint64
	ext   uint64 // fixed live-out contributed by successors outside the SCC
}

type blockAndIndex struct {
	b     *Block
	index int
}

func alternatingOrders(blks []*Block) (entryward, exitward []*Block) {
	n := len(blks)
	switch n {
	case 0:
		return
	case 1:
		return blks, blks
	case 2:
		return []*Block{blks[1], blks[0]}, blks
	}
	inSCC := make(map[int]bool, n)
	for _, b := range blks {
		inSCC[b.ID] = true
	}
	dfsFrom := func(start *Block) []*Block {
		seen := make(map[int]bool, n)
		order := make([]*Block, 0, n)
		stack := []blockAndIndex{{b: start}}
		seen[start.ID] = true
		for len(stack) > 0 {
			top := len(stack) - 1
			x := &stack[top]
			if x.index < len(x.b.Succs) {
				s := x.b.Succs[x.index]
				x.index++
				if inSCC[s.ID] && !seen[s.ID] {
					seen[s.ID] = true
					stack = append(stack, blockAndIndex{b: s})
				}
				continue
			}
			stack = stack[:top]
			order = append(order, x.b)
		}
		return order
	}
	entryward = dfsFrom(blks[0])
	exitward = dfsFrom(entryward[0])
	return
}

// one Gauss-Seidel sweep in the given order; reports whether anything changed
func sweep(order []*Block, in map[int]uint64, inSCC map[int]bool) bool {
	changed := false
	for _, b := range order {
		out := b.ext
		for _, s := range b.Succs {
			if inSCC[s.ID] {
				out |= in[s.ID]
			}
		}
		nv := b.use | (out &^ b.def)
		if nv != in[b.ID] {
			in[b.ID] = nv
			changed = true
		}
	}
	return changed
}

func fixedPoint(blks []*Block, inSCC map[int]bool) map[int]uint64 {
	in := map[int]uint64{}
	order := blks
	for i := 0; i < 10000; i++ {
		if !sweep(order, in, inSCC) {
			return in
		}
	}
	panic("no fixed point")
}

// strongly connected random digraph: Hamiltonian cycle plus extra edges
func randomSCC(r *rand.Rand, n int) []*Block {
	bs := make([]*Block, n)
	for i := range bs {
		bs[i] = &Block{ID: i}
	}
	perm := r.Perm(n)
	for i := 0; i < n; i++ {
		a, b := bs[perm[i]], bs[perm[(i+1)%n]]
		a.Succs = append(a.Succs, b)
	}
	for i := 0; i < n; i++ {
		for k := r.Intn(3); k > 0; k-- {
			bs[i].Succs = append(bs[i].Succs, bs[r.Intn(n)])
		}
	}
	for _, b := range bs {
		b.def = uint64(r.Uint32()) & 0xff
		b.use = uint64(r.Uint32()) & 0xff
		if r.Intn(3) == 0 {
			b.ext = uint64(r.Uint32()) & 0xff
		}
	}
	return bs
}

func main() {
	r := rand.New(rand.NewSource(7))
	const trials = 200000
	hist := map[int]int{}
	notConverged, worst := 0, 0
	over, under := 0, 0
	var missingEx uint64
	var example []*Block

	for t := 0; t < trials; t++ {
		n := 3 + r.Intn(10)
		blks := randomSCC(r, n)
		inSCC := map[int]bool{}
		for _, b := range blks {
			inSCC[b.ID] = true
		}
		want := fixedPoint(blks, inSCC)

		ew, xw := alternatingOrders(blks)
		got := map[int]uint64{}
		passes := 0
		for iter := 0; iter < 64; iter++ {
			order := ew
			if iter&1 == 1 {
				order = xw
			}
			passes++
			if !sweep(order, got, inSCC) {
				break
			}
		}
		hist[passes]++
		if passes > worst {
			worst = passes
			if passes > 3 {
				example = blks
			}
		}
		// what the compiler would produce with its cap of 3
		capped := map[int]uint64{}
		for iter := 0; iter < 3; iter++ {
			order := ew
			if iter&1 == 1 {
				order = xw
			}
			if !sweep(order, capped, inSCC) {
				break
			}
		}
		bad := false
		for id, v := range want {
			if capped[id] != v {
				bad = true
				if capped[id]&^v != 0 {
					over++ // capped has bits the fixed point lacks
				} else {
					under++ // capped is a strict subset: MISSING live values
					if missingEx == 0 {
						missingEx = v &^ capped[id]
					}
				}
				break
			}
		}
		if bad {
			notConverged++
		}
	}

	fmt.Printf("SCCs simulated: %d\n\n", trials)
	fmt.Println("passes needed to reach the fixed point (alternating orders):")
	for p := 1; p <= worst; p++ {
		if hist[p] > 0 {
			fmt.Printf("  %2d passes: %8d  (%5.2f%%)\n", p, hist[p], 100*float64(hist[p])/trials)
		}
	}
	fmt.Printf("\nworst observed: %d passes\n", worst)
	fmt.Printf("SCCs where the compiler's 3-pass cap yields a WRONG result: %d (%.4f%%)\n",
		notConverged, 100*float64(notConverged)/trials)
	fmt.Printf("  of those, UNDER-approximations (missing live values): %d\n", under)
	fmt.Printf("  of those, over-approximations (extra live values):    %d\n", over)
	if missingEx != 0 {
		fmt.Printf("  example missing-bit mask: %08b\n", missingEx)
	}
	if example != nil {
		fmt.Printf("\nexample needing >3 passes: |scc|=%d\n", len(example))
		for _, b := range example {
			fmt.Printf("  b%d -> ", b.ID)
			for _, s := range b.Succs {
				fmt.Printf("b%d ", s.ID)
			}
			fmt.Printf("  def=%02x use=%02x ext=%02x\n", b.def, b.use, b.ext)
		}
	}
}

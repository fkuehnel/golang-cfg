// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Strongly connected component (SCC) detection using Kosaraju-Sharir.
//
// Kosaraju-Sharir is chosen because:
//   - The first DFS pass (postorder) is typically already cached
//   - Straightforward iterative implementation
//   - No auxiliary data on graph nodes required
// Implementation details:
//   - Unreachable blocks are excluded from the result.
//   - Generational memory management is used to avoid bad O(N*D)
//     with N number of blocks and D nested loop depth.
//
// Example:
//
//  Given:  b1 → b2, b2 → [b3, b4], b3 → b2, b4 → b5
//  Result: [[b1], [b2, b3], [b4], [b5]]

// Lesson here is that memory management in recursive SCC
// decompositions is critical to optimal O(n) performance.

// sccWork provides reusable workspace for SCC algorithms.
// Uses generational markers: slot is "set" iff mark[id] == gen.
// Clearing is O(1): just increment the generation.
type sccWork struct {
	// Generation counters
	scopeGen uint32
	seenGen  uint32

	// Per-block markers (indexed by block ID)
	scopeMark []uint32 // in scope if scopeMark[id] == scopeGen
	seenMark  []uint32 // visited if seenMark[id] == seenGen

	// SCC membership (overwritten each pass, no generation needed)
	blockSCC []int

	// Reusable stack
	stack []*Block
}

func newSCCWork(n int) *sccWork {
	return &sccWork{
		scopeGen:  1, // Start at 1 so zero-initialized marks are "unset"
		seenGen:   1,
		scopeMark: make([]uint32, n),
		seenMark:  make([]uint32, n),
		blockSCC:  make([]int, n),
		stack:     make([]*Block, 0, 64),
	}
}

// EntryEdge represents a CFG edge entering an SCC from outside.
type EntryEdge struct {
	From *Block // Source block (outside SCC)
	To   *Block // Target block (inside SCC)
}

// SCC represents a strongly connected component with entry analysis.
type SCC struct {
	Blocks  []*Block    // Blocks in this SCC; leader is Blocks[0]
	Entries []EntryEdge // Edges entering from outside (nil for trivial SCCs)
}

// IsLoop returns true if this SCC represents a loop (more than one block,
// or a single block with a self-loop).
func (s *SCC) IsLoop() bool {
	if len(s.Blocks) > 1 {
		return true
	}
	if len(s.Blocks) == 1 {
		b := s.Blocks[0]
		for _, e := range b.Succs {
			if e.b == b {
				return true // self-loop
			}
		}
	}
	return false
}

// IsReducible returns true if this SCC has a single entry point.
// Trivial SCCs (no loop) are considered reducible.
func (s *SCC) IsReducible() bool {
	if !s.IsLoop() {
		return true
	}
	if s.Entries == nil || len(s.Entries) == 0 {
		return true // No entry info; assume reducible
	}
	// Check all entries target the same block
	first := s.Entries[0].To.ID
	for _, e := range s.Entries[1:] {
		if e.To.ID != first {
			return false
		}
	}
	return true
}

// Header returns the outermost loop header for reducible loops.
// Returns nil for non-loops, irreducible SCCs, or when entry info is unavailable.
func (s *SCC) Header() *Block {
	if !s.IsLoop() || !s.IsReducible() || len(s.Entries) == 0 {
		return nil // caller could use headerByDominance if applicable.
	}
	return s.Entries[0].To
}

// headerByDominance finds the loop header using dominator info.
// This function is used for testing purpose.
// Returns nil if no such block exists (irreducible loop).
func headerByDominance(sdom SparseTree, blocks []*Block) *Block {
	if len(blocks) == 0 {
		return nil
	}
	if len(blocks) == 1 {
		return blocks[0]
	}

	// Build set for quick membership test
	inSCC := make(map[ID]bool, len(blocks))
	for _, b := range blocks {
		inSCC[b.ID] = true
	}

	// Find block that dominates all others
	for _, candidate := range blocks {
		dominatesAll := true
		for _, b := range blocks {
			if b != candidate && !sdom.IsAncestorEq(candidate, b) {
				dominatesAll = false
				break
			}
		}
		if dominatesAll {
			return candidate
		}
	}

	// No single dominator = irreducible
	return nil
}

// EntryTargets returns distinct blocks that receive entry edges.
func (s *SCC) EntryTargets() []*Block {
	if s.Entries == nil {
		if len(s.Blocks) > 0 {
			return []*Block{s.Blocks[0]}
		}
		return nil
	}
	seen := make(map[ID]bool, len(s.Entries))
	targets := make([]*Block, 0, 1)
	for _, e := range s.Entries {
		if !seen[e.To.ID] {
			seen[e.To.ID] = true
			targets = append(targets, e.To)
		}
	}
	return targets
}

// computeSCCs computes all SCCs with entry edge information.
// - Results are in topological order of the condensation DAG.
// - Unreachable blocks are excluded from the result.
func (f *Func) computeSCCs() []SCC {
	po := f.postorder()
	w := newSCCWork(f.NumBlocks())

	// Scope = all reachable blocks
	for _, b := range po {
		w.scopeMark[b.ID] = w.scopeGen
	}

	return w.kosaraju(po)
}

// sccSubgraph computes SCCs within a subgraph, excluding specified block.
// Uses the workspace to avoid O(N) allocations.
func (w *sccWork) sccSubgraph(f *Func, blocks []*Block, exclude *Block) []SCC {
	if len(blocks) == 0 {
		return nil
	}

	// New scope generation = O(1) clear
	w.scopeGen++
	for _, b := range blocks {
		if b != exclude {
			w.scopeMark[b.ID] = w.scopeGen
		}
	}

	// Compute postorder within scope
	po := w.postorder(blocks, exclude)
	if len(po) == 0 {
		return nil
	}

	return w.kosaraju(po)
}

// postorder computes DFS postorder for blocks in current scope.
func (w *sccWork) postorder(blocks []*Block, exclude *Block) []*Block {
	// New seen generation = O(1) clear
	w.seenGen++
	scopeGen := w.scopeGen
	seenGen := w.seenGen

	var po []*Block
	var dfs func(*Block)
	dfs = func(b *Block) {
		if w.scopeMark[b.ID] != scopeGen || w.seenMark[b.ID] == seenGen {
			return
		}
		w.seenMark[b.ID] = seenGen
		for _, e := range b.Succs {
			dfs(e.b)
		}
		po = append(po, b)
	}

	for _, b := range blocks {
		if b != exclude {
			dfs(b)
		}
	}
	return po
}

// kosaraju performs second pass: reverse postorder with DFS on predecessors.
// Results are in topological order.
func (w *sccWork) kosaraju(po []*Block) []SCC {
	// New seen generation = O(1) clear
	w.seenGen++
	scopeGen := w.scopeGen
	seenGen := w.seenGen
	blockSCC := w.blockSCC
	stack := w.stack[:0]

	result := make([]SCC, 0, len(po))
	sccIdx := 0

	for i := len(po) - 1; i >= 0; i-- {
		leader := po[i]
		if w.seenMark[leader.ID] == seenGen {
			continue
		}

		sccIdx++
		scc := make([]*Block, 0, 4)

		// DFS on reverse edges
		stack = append(stack[:0], leader)
		w.seenMark[leader.ID] = seenGen

		for len(stack) > 0 {
			b := stack[len(stack)-1]
			stack = stack[:len(stack)-1]

			scc = append(scc, b)
			blockSCC[b.ID] = sccIdx

			for _, e := range b.Preds {
				pred := e.b
				if w.scopeMark[pred.ID] == scopeGen && w.seenMark[pred.ID] != seenGen {
					w.seenMark[pred.ID] = seenGen
					stack = append(stack, pred)
				}
			}
		}

		// Collect entry edges for non-trivial SCCs
		var entries []EntryEdge
		if len(scc) > 1 {
			for _, b := range scc {
				for _, e := range b.Preds {
					if w.scopeMark[e.b.ID] == scopeGen && blockSCC[e.b.ID] != sccIdx {
						entries = append(entries, EntryEdge{From: e.b, To: b})
					}
				}
			}
		}

		result = append(result, SCC{Blocks: scc, Entries: entries})
	}

	w.stack = stack // Preserve capacity
	return result
}

// This is a convenience wrapper that creates a temporary workspace.
// For repeated calls (like in Bourdoncle decomposition), use sccWork directly.
func sccSubgraph(f *Func, blocks []*Block, exclude *Block) []SCC {
	if len(blocks) == 0 {
		return nil
	}
	w := newSCCWork(f.NumBlocks())
	return w.sccSubgraph(f, blocks, exclude)
}

// sccPartition returns SCCs as [][]*Block for backward compatibility.
func sccPartition(f *Func) [][]*Block {
	sccs := f.sccs()
	result := make([][]*Block, len(sccs))
	for i, scc := range sccs {
		result[i] = scc.Blocks
	}
	return result
}

// sccAlternatingOrdersDFS computes two traversal orders for SCC iteration.
// entryward: DFS postorder from scc[0] (entry)
// exitward: DFS postorder from entryward[0]
func sccAlternatingOrdersDFS(scc []*Block) (entryward, exitward []*Block) {
	n := len(scc)
	switch n {
	case 0:
		return
	case 1:
		entryward, exitward = scc, scc
		return
	case 2:
		entryward = []*Block{scc[1], scc[0]}
		exitward = scc
		return
	}

	// Build membership set for O(1) lookup
	inSCC := make(map[ID]bool, n)
	for _, b := range scc {
		inSCC[b.ID] = true
	}

	// DFS postorder from a starting block, only following edges within SCC
	dfsFrom := func(start *Block) []*Block {
		seen := make(map[ID]bool, n)
		order := make([]*Block, 0, n)
		stack := make([]blockAndIndex, 0, n)

		seen[start.ID] = true
		stack = append(stack, blockAndIndex{b: start})

		for len(stack) > 0 {
			top := len(stack) - 1
			x := &stack[top]

			if x.index < len(x.b.Succs) {
				succ := x.b.Succs[x.index].b
				x.index++
				if inSCC[succ.ID] && !seen[succ.ID] {
					seen[succ.ID] = true
					stack = append(stack, blockAndIndex{b: succ})
				}
				continue
			}

			// All successors visited, emit in postorder
			stack = stack[:top]
			order = append(order, x.b)
		}
		return order
	}

	// entryward: DFS postorder from scc[0]
	entryward = dfsFrom(scc[0])
	// exitward: DFS postorder from entryward[0]
	exitward = dfsFrom(entryward[0])
	return
}

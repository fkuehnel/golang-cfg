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
//
// Example:
//
//  Given:  b1 → b2, b2 → [b3, b4], b3 → b2, b4 → b5
//  Result: [[b1], [b2, b3], [b4], [b5]]

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
// Results are in topological order of the condensation DAG.
// Unreachable blocks are excluded from the result.
func (f *Func) computeSCCs() []SCC {
	po := f.postorder()

	// exclude dead (non-reachable) blocks
	scope := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(scope)
	for _, b := range po {
		scope[b.ID] = true
	}

	return kosarajuSCCs(f, po, scope)
}

// sccSubgraph computes SCCs within a subgraph, excluding specified block.
// Used for recursive Bourdoncle decomposition.
func sccSubgraph(f *Func, blocks []*Block, exclude *Block) []SCC {
	if len(blocks) == 0 {
		return nil
	}

	// Build scope: valid blocks in subgraph
	scope := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(scope)
	for _, b := range blocks {
		scope[b.ID] = (b != exclude)
	}

	// Compute local postorder (handles disconnected subgraphs)
	po := subgraphPostorder(f, blocks, exclude, scope)
	if len(po) == 0 {
		return nil
	}

	return kosarajuSCCs(f, po, scope)
}

// subgraphPostorder computes postorder for blocks within scope.
func subgraphPostorder(f *Func, blocks []*Block, exclude *Block, scope []bool) []*Block {
	visited := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(visited)

	// Ensure visited is false for all blocks in scope; cache may return unzeroed memory.
	for _, b := range blocks {
		if b != exclude && scope[b.ID] {
			visited[b.ID] = false
		}
	}

	var po []*Block
	var dfs func(*Block)
	dfs = func(b *Block) {
		if !scope[b.ID] || visited[b.ID] {
			return
		}
		visited[b.ID] = true
		for _, e := range b.Succs {
			dfs(e.b)
		}
		po = append(po, b)
	}

	// Start from each block to handle disconnected components
	for _, b := range blocks {
		if b != exclude {
			dfs(b)
		}
	}
	return po
}

// kosarajuSCCs performs Kosaraju's second pass: reverse postorder traversal
// with BFS on reverse edges to collect SCCs.
//
// If scope is nil, all predecessors are considered valid (fast path for top-level).
// If scope is non-nil, only predecessors where scope[id]==true are followed.
func kosarajuSCCs(f *Func, po []*Block, scope []bool) []SCC {
	n := f.NumBlocks()

	seen := f.Cache.allocBoolSlice(n)
	defer f.Cache.freeBoolSlice(seen)
	queue := f.Cache.allocBlockSlice(len(po))
	defer f.Cache.freeBlockSlice(queue)

	// SCC membership for entry edge computation. When scope != nil, we only
	// consult blockSCC for predecessors in scope; by topological ordering, those
	// predecessors have already been assigned.
	var blockSCC []int
	if scope == nil {
		blockSCC = make([]int, n)
	} else {
		blockSCC = f.Cache.allocIntSlice(n)
		defer f.Cache.freeIntSlice(blockSCC)
	}

	result := make([]SCC, 0, len(po))
	sccIdx := 0
	queue = queue[:0]

	// Process in reverse postorder
	for i := len(po) - 1; i >= 0; i-- {
		leader := po[i]
		if seen[leader.ID] {
			continue
		}

		sccIdx++
		scc := make([]*Block, 0, 4)
		queue = append(queue[:0], leader)
		seen[leader.ID] = true

		// BFS on reverse edges
		for len(queue) > 0 {
			b := queue[0]
			queue = queue[1:]
			scc = append(scc, b)
			blockSCC[b.ID] = sccIdx

			for _, e := range b.Preds {
				pred := e.b
				if (scope == nil || scope[pred.ID]) && !seen[pred.ID] {
					seen[pred.ID] = true
					queue = append(queue, pred)
				}
			}
		}

		// Collect entry edges for non-trivial SCCs
		var entries []EntryEdge
		if len(scc) > 1 {
			for _, b := range scc {
				for _, e := range b.Preds {
					if (scope == nil || scope[e.b.ID]) && blockSCC[e.b.ID] != sccIdx {
						entries = append(entries, EntryEdge{From: e.b, To: b})
					}
				}
			}
		}

		result = append(result, SCC{Blocks: scc, Entries: entries})
	}

	return result
}

// kosarajuSCCs performs Kosaraju's second pass: reverse postorder traversal
// with DFS on reverse edges to collect SCCs. Each SCC is order
// by preorder on reverse graph (Preds).
//
// If scope is nil, all predecessors are considered valid (fast path for top-level).
// If scope is non-nil, only predecessors where scope[id]==true are followed.
func kosarajuSCCs(f *Func, po []*Block, scope []bool) []SCC {
	n := f.NumBlocks()

	seen := f.Cache.allocBoolSlice(n)
	defer f.Cache.freeBoolSlice(seen)
	stack := f.Cache.allocBlockSlice(len(po))
	defer f.Cache.freeBlockSlice(stack)

	// SCC membership for entry edge computation
	var blockSCC []int
	if scope == nil {
		blockSCC = make([]int, n)
	} else {
		blockSCC = f.Cache.allocIntSlice(n)
		defer f.Cache.freeIntSlice(blockSCC)
	}

	result := make([]SCC, 0, len(po))
	sccIdx := 0

	// Process in reverse postorder
	for i := len(po) - 1; i >= 0; i-- {
		leader := po[i]
		if seen[leader.ID] {
			continue
		}

		sccIdx++
		scc := make([]*Block, 0, 4)
		stack = stack[:1]
		stack[0] = leader
		seen[leader.ID] = true

		// DFS on reverse edges
		for len(stack) > 0 {
			// Pop from end - O(1)
			top := len(stack) - 1
			b := stack[top]
			stack = stack[:top]

			scc = append(scc, b)
			blockSCC[b.ID] = sccIdx

			for _, e := range b.Preds {
				pred := e.b
				if (scope == nil || scope[pred.ID]) && !seen[pred.ID] {
					seen[pred.ID] = true
					stack = append(stack, pred)
				}
			}
		}

		// Collect entry edges for non-trivial SCCs
		var entries []EntryEdge
		if len(scc) > 1 {
			for _, b := range scc {
				for _, e := range b.Preds {
					if (scope == nil || scope[e.b.ID]) && blockSCC[e.b.ID] != sccIdx {
						entries = append(entries, EntryEdge{From: e.b, To: b})
					}
				}
			}
		}

		result = append(result, SCC{Blocks: scc, Entries: entries})
	}

	return result
}

// hasSelfLoop returns true if any block in the SCC has a self-loop edge.
func hasSelfLoop(blocks []*Block) bool {
	for _, b := range blocks {
		for _, e := range b.Succs {
			if e.b == b {
				return true
			}
		}
	}
	return false
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

// sccAlternatingOrdersBFS computes two traversal orders for SCC iteration.
// entryward: scc blocks in reverse order
// exitward: reversed BFS from scc[n-1] (last block)
func sccAlternatingOrdersBFS(scc []*Block) (entryward, exitward []*Block) {
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

	// BFS from a starting block, only following edges within SCC
	bfsFrom := func(start *Block) []*Block {
		seen := make(map[ID]bool, len(scc))
		order := make([]*Block, 0, len(scc))
		queue := []*Block{start}
		seen[start.ID] = true

		for len(queue) > 0 {
			b := queue[0]
			queue = queue[1:]
			order = append(order, b)

			for _, e := range b.Succs {
				succ := e.b
				if inSCC[succ.ID] && !seen[succ.ID] {
					seen[succ.ID] = true
					queue = append(queue, succ)
				}
			}
		}
		return order
	}

	// entryward: scc blocks reversed
	entryward = make([]*Block, n)
	for i, b := range scc {
		entryward[n-1-i] = b
	}
	// exitward: BFS from scc[n-1], then reversed
	bfs1 := bfsFrom(scc[n-1])
	exitward = make([]*Block, n)
	for i, b := range bfs1 {
		exitward[n-1-i] = b
	}
	return
}

// sccAlternatingOrdersDFS computes two traversal orders for SCC iteration.
// entryward: reversed DFS postorder from scc[0] (entry)
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

	// entryward: DFS postorder from scc[0], then reversed
	dfs1 := dfsFrom(scc[0])
	entryward = make([]*Block, n)
	for i, b := range dfs1 {
		entryward[n-1-i] = b
	}
	// exitward: DFS postorder from entryward[0]
	exitward = dfsFrom(entryward[0])
	return
}
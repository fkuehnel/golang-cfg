// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "iter"

// This file implements strongly connected component (SCC) detection for
// control-flow graphs using the Kosaraju-Sharir algorithm.
//
// Kosaraju-Sharir was chosen over Tarjan's single-pass algorithm because it is
// straightforward to implement iteratively and requires no auxiliary data on
// graph nodes. Additionally, the first DFS pass (postorder) is typically already
// computed and cached, making this choice effectively free.
//
// sccPartition returns the strongly connected components of f's control-flow
// graph, topologically sorted by the kernel DAG. Each SCC corresponds to a loop
// (or trivial single-block component) in f.
//
// Properties:
//   - The first SCC contains only the entry block.
//   - Unreachable blocks are excluded from the result.
//   - The topological order of the kernel DAG may not be unique, but this does
//     not affect correctness for live range computation.
//   - Block order within each SCC is unspecified.
//
// The iterator pattern avoids allocating the result slice when callers
// only need a single traversal.
//
// Example:
//
//	Given:  b1 → b2, b2 → [b3, b4], b3 → b2, b4 → b5
//	Result: [[b1], [b2, b3], [b4], [b5]]
//
// The second pass uses BFS with reversed edges for simplicity.
func (f *Func) SCCs() iter.Seq[[]*Block] {
	return func(yield func([]*Block) bool) {
		// First DFS pass: compute postorder on original edges.
		// The last element is the function entry block.
		po := f.postorder()

		// Track visited blocks and filter to reachable only.
		seen := make([]bool, f.NumBlocks())
		reachable := make([]bool, f.NumBlocks())
		for _, b := range po {
			reachable[b.ID] = true
		}

		// Second pass: traverse reversed edges in reverse postorder.
		// Each connected component found is an SCC.
		queue := make([]*Block, 0, len(po))

		for i := len(po) - 1; i >= 0; i-- {
			leader := po[i]
			if seen[leader.ID] {
				continue
			}

			// BFS to find all blocks in this SCC.
			scc := make([]*Block, 0, 4)
			queue = append(queue, leader)
			seen[leader.ID] = true

			for len(queue) > 0 {
				b := queue[0]
				queue = queue[1:]
				scc = append(scc, b)

				for _, e := range b.Preds {
					pred := e.b
					if reachable[pred.ID] && !seen[pred.ID] {
						seen[pred.ID] = true
						queue = append(queue, pred)
					}
				}
			}

			if !yield(scc) {
				return
			}
		}
	}
}

// sccPartition returns all SCCs as a slice for callers that need random access.
// Prefer [Func.SCCs] when iterating once.
func sccPartition(f *Func) [][]*Block {
	var result [][]*Block
	for scc := range f.SCCs() {
		result = append(result, scc)
	}
	return result
}

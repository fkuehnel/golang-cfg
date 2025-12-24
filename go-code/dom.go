/ Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file contains code to compute the dominator tree
// of a control-flow graph.

// postorder computes a postorder traversal ordering for the
// basic blocks in f. Unreachable blocks will not appear.
func postorder(f *Func) []*Block {
	return postorderWithNumbering(f, nil)
}

type blockAndIndex struct {
	b     *Block
	index int // index is the number of successor edges of b that have already been explored.
}

// postorderWithNumbering provides a DFS postordering.
// This seems to make loop-finding more robust.
func postorderWithNumbering(f *Func, ponums []int32) []*Block {
	valid := make([]bool, f.NumBlocks())
	for i := 0; i < len(valid); i++ {
		valid[i] = true
	}
	return poWithNumberingForValidBlocks(f.Entry, valid, ponums)
}

func poWithNumberingForValidBlocks(entry *Block, valid []bool, ponums []int32) []*Block {
	f := entry.Func
	if len(valid) != f.NumBlocks() {
		f.Fatalf("length of valid blocks is expected to be %d", f.NumBlocks())
	}
	seen := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(seen)

	// result ordering
	order := make([]*Block, 0, len(f.Blocks))

	// stack of blocks and next child to visit
	// A constant bound allows this to be stack-allocated. 32 is
	// enough to cover almost every postorderWithNumbering call.
	s := make([]blockAndIndex, 0, 32)
	s = append(s, blockAndIndex{b: entry})
	seen[entry.ID] = true
	for len(s) > 0 {
		tos := len(s) - 1
		x := s[tos]
		b := x.b
		if i := x.index; i < len(b.Succs) {
			s[tos].index++
			bb := b.Succs[i].Block()
			if valid[bb.ID] && !seen[bb.ID] {
				seen[bb.ID] = true
				s = append(s, blockAndIndex{b: bb})
			}
			continue
		}
		s = s[:tos]
		if ponums != nil {
			ponums[b.ID] = int32(len(order))
		}
		order = append(order, b)
	}
	return order
}

// intersect finds the closest dominator of both b and c.
// It requires a postorder numbering of all the blocks.
func intersect(b, c *Block, postnum []int, idom []*Block) *Block {
	// TODO: This loop is O(n^2). It used to be used in nilcheck,
	// see BenchmarkNilCheckDeep*.
	for b != c {
		if postnum[b.ID] < postnum[c.ID] {
			b = idom[b.ID]
		} else {
			c = idom[c.ID]
		}
	}
	return b
}

// finds postorder and reverse postorder within SCC.
func sccAlternatingOrders(scc []*Block) (exitward, entryward []*Block) {
	if len(scc) == 2 {
		// Trivial case: just swap order
		return scc, []*Block{scc[1], scc[0]}
	}
	entry := scc[0]
	f := entry.Func

	// limit the graph to only blocks within the SCC
	valid := f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(valid)
	for _, b := range scc {
		valid[b.ID] = true
	}
	exitward = poWithNumberingForValidBlocks(entry, valid, nil)
	entryward = poWithNumberingForValidBlocks(exitward[0], valid, nil)

	return exitward, entryward
}

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

// sccFunc is the signature for backward-compatible SCC partition functions.
type sccFunc func(f *Func) [][]*Block

// verifySccPartition verifies that the SCCs are topologically sorted
// and each SCC contains the proper set of nodes.
func verifySccPartition(t *testing.T, fut fun, sccFn sccFunc, sccPartition [][]string) {
	t.Helper()
	blockNames := map[*Block]string{}
	for n, b := range fut.blocks {
		blockNames[b] = n
	}

	calcScc := sccFn(fut.f)
	expectedNscc := len(sccPartition)
	actualNscc := len(calcScc)
	if actualNscc != expectedNscc {
		t.Errorf("expected %d SCC kernels, found %d", expectedNscc, actualNscc)
	}

	for n, scc := range calcScc {
		if n >= len(sccPartition) {
			break
		}
		expectedScc := sccPartition[n]
		nc := len(scc)
		if nc != len(expectedScc) {
			t.Errorf("SCC %d: expected %v nodes, found %v", n, expectedScc, blockNamesToSlice(scc, blockNames))
		}

		// Verify nodes in this SCC match expected
		nodeSet := make(map[string]bool)
		for _, b := range scc {
			nodeSet[blockNames[b]] = false
		}

		for _, expectedNode := range expectedScc {
			if val, ok := nodeSet[expectedNode]; !ok {
				t.Errorf("SCC %d: expected node %s not found in %v", n, expectedNode, blockNamesToSlice(scc, blockNames))
			} else if val {
				t.Errorf("SCC %d: duplicate expected node %s", n, expectedNode)
			}
			nodeSet[expectedNode] = true
		}

		for k, v := range nodeSet {
			if !v {
				t.Errorf("SCC %d: unexpected block %s", n, k)
			}
		}
	}
}

// blockNames returns block names for debugging
func blockNames(blocks []*Block, nameMap map[string]*Block) []string {
	// Invert the map
	inv := make(map[*Block]string)
	for name, b := range nameMap {
		inv[b] = name
	}
	names := make([]string, len(blocks))
	for i, b := range blocks {
		if name, ok := inv[b]; ok {
			names[i] = name
		} else {
			names[i] = b.String()
		}
	}
	return names
}

func blockNamesToSlice(blocks []*Block, names map[*Block]string) []string {
	result := make([]string, len(blocks))
	for i, b := range blocks {
		result[i] = names[b]
	}
	return result
}

func TestSCCPartitionLinear(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Goto("1")),
		Bloc("1",
			Goto("2")),
		Bloc("2",
			Goto("3")),
		Bloc("3",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"1"},
		{"2"},
		{"3"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)

	// Verify no loops detected
	for _, scc := range fun.f.sccs() {
		if scc.IsLoop() {
			t.Errorf("linear CFG should have no loops, found loop at %v", scc.Blocks[0])
		}
	}
}

func TestSCCPartitionOneLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			If("p", "a", "b")),
		Bloc("a",
			Goto("c")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			If("p", "b", "exit")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"a"},
		{"b", "c"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)

	// Note: This loop has TWO entry edges (entry→b and a→c) targeting
	// different blocks, making it irreducible by strict single-entry definition.
	// The SCC {b,c} is still detected, but has no unique header.
	for _, scc := range fun.f.sccs() {
		if scc.IsLoop() {
			// Two entry targets = irreducible
			if scc.IsReducible() {
				t.Error("expected irreducible loop (two entry targets)")
			}
			if scc.Header() != nil {
				t.Errorf("expected nil header for irreducible loop, got %v", scc.Header())
			}
			// But we should have 2 entry edges
			if len(scc.Entries) != 2 {
				t.Errorf("expected 2 entry edges, got %d", len(scc.Entries))
			}
		}
	}
}

func TestSccPartitionReducibleLoop(t *testing.T) {
	c := testConfig(t)
	// A proper reducible loop with single entry point
	//
	//   entry -> header -> body
	//               ^        |
	//               +--------+
	//               |
	//               v
	//              exit
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("header")),
		Bloc("header",
			If("p", "body", "exit")),
		Bloc("body",
			Goto("header")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"header", "body"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)

	// This loop has ONE entry edge (entry→header), making it reducible
	for _, scc := range fun.f.sccs() {
		if scc.IsLoop() {
			if !scc.IsReducible() {
				t.Error("expected reducible loop")
			}
			header := scc.Header()
			if header == nil {
				t.Fatal("expected non-nil header for reducible loop")
			}
			if header != fun.blocks["header"] {
				t.Errorf("expected header 'header', got %v", header)
			}
			if len(scc.Entries) != 1 {
				t.Errorf("expected 1 entry edge, got %d", len(scc.Entries))
			}
		}
	}
}

func TestSCCPartitionInfiniteLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("a")),
		Bloc("a",
			Goto("b")),
		Bloc("b",
			Goto("a")),
	)

	expectedSccPartition := [][]string{
		{"entry"},
		{"b", "a"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSCCPartitionDeadCode(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 0, nil),
			If("p", "b3", "b5")),
		Bloc("b2", Exit("mem")),
		Bloc("b3", Goto("b2")),
		Bloc("b4", Goto("b2")), // unreachable
		Bloc("b5", Goto("b2")))

	// Topological order not unique: b3 and b5 interchangeable
	expectedSccPartition := [][]string{
		{"entry"},
		{"b5"},
		{"b3"},
		{"b2"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSccPartitionTricky(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			If("p", "6", "8")),
		Bloc("1",
			If("p", "exit", "5")),
		Bloc("2",
			If("p", "1", "3")),
		Bloc("3",
			If("p", "5", "2")),
		Bloc("4",
			If("p", "2", "3")),
		Bloc("5",
			Goto("4")),
		Bloc("6",
			If("p", "7", "4")),
		Bloc("7",
			If("p", "6", "8")),
		Bloc("8",
			If("p", "10", "9")),
		Bloc("9",
			Goto("11")),
		Bloc("10",
			If("p", "11", "4")),
		Bloc("11",
			Goto("8")),
		Bloc("exit",
			Exit("mem")))

	expectedSccPartition := [][]string{
		{"entry"},
		{"6", "7"},
		{"8", "9", "10", "11"},
		{"1", "2", "3", "4", "5"},
		{"exit"},
	}

	CheckFunc(fun.f)
	verifySccPartition(t, fun, sccPartition, expectedSccPartition)
}

func TestSCCEntryEdges(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("header")),
		Bloc("header",
			If("p", "body", "exit")),
		Bloc("body",
			Goto("header")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)

	sccs := fun.f.sccs()

	// Find the loop SCC
	var loopSCC *SCC
	for i := range sccs {
		if sccs[i].IsLoop() {
			loopSCC = &sccs[i]
			break
		}
	}

	if loopSCC == nil {
		t.Fatal("expected to find a loop SCC")
	}

	// Should have exactly one entry edge
	if len(loopSCC.Entries) != 1 {
		t.Errorf("expected 1 entry edge, got %d", len(loopSCC.Entries))
	}

	// Entry should be from "entry" to "header"
	if loopSCC.Entries[0].From != fun.blocks["entry"] {
		t.Errorf("expected entry from 'entry', got %v", loopSCC.Entries[0].From)
	}
	if loopSCC.Entries[0].To != fun.blocks["header"] {
		t.Errorf("expected entry to 'header', got %v", loopSCC.Entries[0].To)
	}

	// Should be reducible
	if !loopSCC.IsReducible() {
		t.Error("expected reducible loop")
	}

	// Header should be "header"
	if loopSCC.Header() != fun.blocks["header"] {
		t.Errorf("expected header 'header', got %v", loopSCC.Header())
	}
}

func TestSCCIrreducible(t *testing.T) {
	c := testConfig(t)
	// Classic irreducible loop: two entry points
	//
	//     entry
	//    /     \
	//   v       v
	//   a  <->  b
	//    \     /
	//     v   v
	//      exit
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			If("p", "a", "b")),
		Bloc("a",
			If("p", "b", "exit")),
		Bloc("b",
			If("p", "a", "exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)

	sccs := fun.f.sccs()

	// Find the loop SCC (a and b)
	var loopSCC *SCC
	for i := range sccs {
		if sccs[i].IsLoop() {
			loopSCC = &sccs[i]
			break
		}
	}

	if loopSCC == nil {
		t.Fatal("expected to find a loop SCC")
	}

	// Should have two entry edges (entry->a and entry->b)
	if len(loopSCC.Entries) != 2 {
		t.Errorf("expected 2 entry edges, got %d", len(loopSCC.Entries))
	}

	// Should be irreducible (multiple entry targets)
	if loopSCC.IsReducible() {
		t.Error("expected irreducible loop")
	}

	// Header should be nil for irreducible
	if loopSCC.Header() != nil {
		t.Errorf("expected nil header for irreducible loop, got %v", loopSCC.Header())
	}

	// Should have two entry targets
	targets := loopSCC.EntryTargets()
	if len(targets) != 2 {
		t.Errorf("expected 2 entry targets, got %d", len(targets))
	}
}

func TestSCCSelfLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("loop")),
		Bloc("loop",
			If("p", "loop", "exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)

	// Find the self-loop SCC
	var selfLoopSCC *SCC
	for _, scc := range fun.f.sccs() {
		if len(scc.Blocks) == 1 && scc.Blocks[0] == fun.blocks["loop"] {
			selfLoopSCC = &scc
			break
		}
	}

	if selfLoopSCC == nil {
		t.Fatal("expected to find self-loop SCC")
	}

	if !selfLoopSCC.IsLoop() {
		t.Error("self-loop should be detected as a loop")
	}
}

func TestSCCNestedLoops(t *testing.T) {
	c := testConfig(t)
	// Outer loop with inner loop
	//
	//   entry -> outer_header -> inner_header -> inner_body -> inner_header
	//                  ^                              |
	//                  |                              v
	//                  +--------- outer_body <--------+
	//                  |
	//                  v
	//                exit
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("outer_header")),
		Bloc("outer_header",
			If("p", "inner_header", "exit")),
		Bloc("inner_header",
			If("p", "inner_body", "outer_body")),
		Bloc("inner_body",
			Goto("inner_header")),
		Bloc("outer_body",
			Goto("outer_header")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)

	loopCount := 0
	for _, scc := range fun.f.sccs() {
		if scc.IsLoop() {
			loopCount++
			if !scc.IsReducible() {
				t.Error("all loops should be reducible")
			}
		}
	}

	// At top level, we see one big SCC containing all loop blocks
	// Nested structure requires Bourdoncle decomposition.
	if loopCount != 1 {
		t.Errorf("expected 1 top-level loop SCC, got %d", loopCount)
	}
}

func TestSCCSubgraphRecursive(t *testing.T) {
	c := testConfig(t)
	// Nested loop structure for Bourdoncle decomposition test
	//
	//   entry -> outer_header -> inner_header -> inner_body
	//                  ^               |              |
	//                  |               v              v
	//                  +-------- outer_body <---------+
	//                  |
	//                  v
	//                exit
	//
	// Top-level SCC: {outer_header, inner_header, inner_body, outer_body}
	// After removing outer_header:
	//   - Sub-SCC: {inner_header, inner_body} (inner loop)
	//   - Sub-SCC: {outer_body} (trivial)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("outer_header")),
		Bloc("outer_header",
			If("p", "inner_header", "exit")),
		Bloc("inner_header",
			If("p", "inner_body", "outer_body")),
		Bloc("inner_body",
			Goto("inner_header")),
		Bloc("outer_body",
			Goto("outer_header")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)

	// Step 1: Get top-level SCCs
	sccs := fun.f.sccs()

	// Find the loop SCC
	var loopSCC *SCC
	for i := range sccs {
		if sccs[i].IsLoop() {
			loopSCC = &sccs[i]
			break
		}
	}
	if loopSCC == nil {
		t.Fatal("expected to find a loop SCC")
	}

	// Verify it contains all 4 loop blocks
	if len(loopSCC.Blocks) != 4 {
		t.Errorf("expected 4 blocks in top-level SCC, got %d", len(loopSCC.Blocks))
	}

	// Step 2: Get header - for top-level SCC, Header() works (has entry edges)
	header := loopSCC.Header()
	if header == nil {
		t.Fatal("expected non-nil header for top-level loop")
	}
	if header != fun.blocks["outer_header"] {
		t.Errorf("expected outer_header as header, got %v", header)
	}

	// Verify entry edges exist
	if len(loopSCC.Entries) == 0 {
		t.Error("expected entry edges for top-level SCC")
	}

	// Step 3: Call sccSubgraph excluding the header (Bourdoncle step)
	subSccs := sccSubgraph(fun.f, loopSCC.Blocks, header)

	// Should get 2 sub-SCCs:
	// - {inner_header, inner_body} - the inner loop
	// - {outer_body} - trivial SCC
	if len(subSccs) != 2 {
		t.Errorf("expected 2 sub-SCCs after removing header, got %d", len(subSccs))
		for i, sub := range subSccs {
			t.Logf("  sub-SCC %d: %v", i, blockNames(sub.Blocks, fun.blocks))
		}
	}

	// Find the inner loop sub-SCC
	var innerLoopSCC *SCC
	var trivialSCC *SCC
	for i := range subSccs {
		if subSccs[i].IsLoop() {
			innerLoopSCC = &subSccs[i]
		} else {
			trivialSCC = &subSccs[i]
		}
	}

	if innerLoopSCC == nil {
		t.Fatal("expected to find inner loop sub-SCC")
	}
	if trivialSCC == nil {
		t.Fatal("expected to find trivial sub-SCC")
	}

	// Verify inner loop has 2 blocks
	if len(innerLoopSCC.Blocks) != 2 {
		t.Errorf("expected 2 blocks in inner loop, got %d", len(innerLoopSCC.Blocks))
	}

	// Verify inner loop is reducible
	if !innerLoopSCC.IsReducible() {
		t.Error("inner loop should be reducible")
	}

	// For nested loops, Header() returns nil (no entry edges from excluded scope)
	// Must use headerByDominance instead
	if innerLoopSCC.Header() != nil {
		t.Error("expected Header() to return nil for nested loop (no entry edges)")
	}

	// Use headerByDominance to find inner loop header
	sdom := fun.f.Sdom()
	innerHeader := headerByDominance(sdom, innerLoopSCC.Blocks)
	if innerHeader == nil {
		t.Fatal("expected headerByDominance to find inner loop header")
	}
	if innerHeader != fun.blocks["inner_header"] {
		t.Errorf("expected inner_header as inner loop header, got %v", innerHeader)
	}

	// Verify trivial SCC is outer_body
	if len(trivialSCC.Blocks) != 1 {
		t.Errorf("expected 1 block in trivial SCC, got %d", len(trivialSCC.Blocks))
	}
	if trivialSCC.Blocks[0] != fun.blocks["outer_body"] {
		t.Errorf("expected outer_body in trivial SCC, got %v", trivialSCC.Blocks[0])
	}

	// Step 4: Recurse into inner loop - remove inner_header
	innerSubSccs := sccSubgraph(fun.f, innerLoopSCC.Blocks, innerHeader)

	// Should get 1 trivial SCC: {inner_body}
	if len(innerSubSccs) != 1 {
		t.Errorf("expected 1 sub-SCC after removing inner header, got %d", len(innerSubSccs))
	}
	if innerSubSccs[0].IsLoop() {
		t.Error("inner_body alone should not be a loop")
	}
	if len(innerSubSccs[0].Blocks) != 1 || innerSubSccs[0].Blocks[0] != fun.blocks["inner_body"] {
		t.Errorf("expected inner_body in final sub-SCC")
	}
}

func TestSCCSubgraphDisconnected(t *testing.T) {
	c := testConfig(t)
	// Test that sccSubgraph handles disconnected subgraphs
	// (can happen when header removal splits the graph)
	//
	//   entry -> header -> a -> b
	//               |      ^    |
	//               v      |    v
	//               c -----+    exit
	//
	// SCC: {header, a, b, c}
	// After removing header: {a, b, c}
	//   - {a, b} forms a loop (a -> b -> a)
	//   - {c} is trivial (c -> a, but a is in different SCC)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("header")),
		Bloc("header",
			If("p", "a", "c")),
		Bloc("a",
			Goto("b")),
		Bloc("b",
			If("p", "a", "exit")),
		Bloc("c",
			Goto("a")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)

	// Get the loop SCC
	sccs := fun.f.sccs()
	var loopSCC *SCC
	for i := range sccs {
		if sccs[i].IsLoop() {
			loopSCC = &sccs[i]
			break
		}
	}
	if loopSCC == nil {
		t.Fatal("expected to find a loop SCC")
	}

	// Top-level header via Header()
	header := loopSCC.Header()
	if header == nil {
		// Use headerByDominance as fallback
		sdom := fun.f.Sdom()
		header = headerByDominance(sdom, loopSCC.Blocks)
	}
	if header == nil {
		t.Fatal("could not determine loop header")
	}

	// Remove header and get sub-SCCs
	subSccs := sccSubgraph(fun.f, loopSCC.Blocks, header)

	// Should handle the disconnected components correctly
	// All remaining blocks should be accounted for
	totalBlocks := 0
	for _, sub := range subSccs {
		totalBlocks += len(sub.Blocks)
	}

	expectedBlocks := len(loopSCC.Blocks) - 1 // minus header
	if totalBlocks != expectedBlocks {
		t.Errorf("expected %d blocks in sub-SCCs, got %d", expectedBlocks, totalBlocks)
	}

	// Verify we can find headers for any nested loops using headerByDominance
	sdom := fun.f.Sdom()
	for i := range subSccs {
		if subSccs[i].IsLoop() {
			h := headerByDominance(sdom, subSccs[i].Blocks)
			if h == nil {
				t.Errorf("sub-SCC %d is a loop but headerByDominance returned nil", i)
			}
		}
	}
}

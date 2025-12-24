// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
	"testing"
)

// =============================================================================
// ACYCLIC CFG BUILDERS
// =============================================================================

// buildLinearChain creates an acyclic CFG: entry -> b1 -> b2 -> ... -> bN -> exit
// Each block has values that create cross-block liveness dependencies.
func buildLinearChain(tb testing.TB, numBlocks int) *Func {
	c := testConfig(tb)
	intType := c.config.Types.Int64

	// We'll build: entry -> b0 -> b1 -> ... -> b{numBlocks-1} -> exit
	// Each block defines a value and uses values from previous blocks
	// to create realistic liveness ranges.

	blocs := make([]bloc, 0, numBlocks+2)

	// Entry block: initialize base values
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
		Valu("v_base", OpConst64, intType, 42, nil),
		Goto("b0")))

	// Chain of body blocks
	for i := 0; i < numBlocks; i++ {
		blockName := fmt.Sprintf("b%d", i)
		nextBlock := fmt.Sprintf("b%d", i+1)
		if i == numBlocks-1 {
			nextBlock = "exit"
		}

		// Current block's value name
		currVal := fmt.Sprintf("v%d", i)

		// Reference a value from earlier block to create liveness range
		// Use v_base for first few, then reference earlier computed values
		var prevVal string
		if i < 3 {
			prevVal = "v_base"
		} else {
			// Reference a value from ~3 blocks back (creates longer live ranges)
			prevVal = fmt.Sprintf("v%d", i-3)
		}

		blocs = append(blocs, Bloc(blockName,
			// Define a new value using a previous value
			Valu(currVal, OpAdd64, intType, 0, nil, prevVal, "v_base"),
			Goto(nextBlock)))
	}

	// Exit block: use the last computed value
	lastVal := fmt.Sprintf("v%d", numBlocks-1)
	blocs = append(blocs, Bloc("exit",
		Valu("retval", OpAdd64, intType, 0, nil, lastVal, "v_base"),
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// buildLinearChainDense creates a chain where each block has multiple values
// and uses values from several previous blocks (denser liveness).
func buildLinearChainDense(tb testing.TB, numBlocks, valsPerBlock int) *Func {
	c := testConfig(tb)
	intType := c.config.Types.Int64

	blocs := make([]bloc, 0, numBlocks+2)

	// Track all value names for cross-references
	allVals := []string{"v_base"}

	// Entry block
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
		Valu("v_base", OpConst64, intType, 1, nil),
		Goto("b0")))

	// Body blocks
	for i := 0; i < numBlocks; i++ {
		blockName := fmt.Sprintf("b%d", i)
		nextBlock := fmt.Sprintf("b%d", i+1)
		if i == numBlocks-1 {
			nextBlock = "exit"
		}

		// Build values for this block
		var blockContents []any
		for j := 0; j < valsPerBlock; j++ {
			valName := fmt.Sprintf("v%d_%d", i, j)

			// Pick two operands from earlier values
			op1Idx := len(allVals) - 1
			op2Idx := 0
			if len(allVals) > 5 {
				op2Idx = len(allVals) - 5 // Reference value from ~5 values back
			}

			blockContents = append(blockContents,
				Valu(valName, OpAdd64, intType, 0, nil, allVals[op1Idx], allVals[op2Idx]))
			allVals = append(allVals, valName)
		}
		blockContents = append(blockContents, Goto(nextBlock))

		blocs = append(blocs, Bloc(blockName, blockContents...))
	}

	// Exit
	lastVal := allVals[len(allVals)-1]
	blocs = append(blocs, Bloc("exit",
		Valu("final", OpAdd64, intType, 0, nil, lastVal, "v_base"),
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// =============================================================================
// SIMPLE LOOP CFG BUILDERS
// =============================================================================

// buildSimpleLoop creates: entry -> header <-> body -> exit
//
//	^         |
//	+---------+ (back edge)
func buildSimpleLoop(tb testing.TB, bodyBlocks int) *Func {
	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	blocs := make([]bloc, 0, bodyBlocks+3)

	// Entry
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("limit", OpConst64, intType, 100, nil),
		Goto("header")))

	// Loop header with phi
	blocs = append(blocs, Bloc("header",
		Valu("i", OpPhi, intType, 0, nil, "zero", "i_inc"),
		Valu("cmp", OpLess64, boolType, 0, nil, "i", "limit"),
		If("cmp", "body0", "exit")))

	// Loop body blocks
	prevVal := "i"
	for j := 0; j < bodyBlocks; j++ {
		blockName := fmt.Sprintf("body%d", j)
		nextBlock := fmt.Sprintf("body%d", j+1)

		currVal := fmt.Sprintf("tmp%d", j)

		if j == bodyBlocks-1 {
			// Last body block: increment and jump back to header
			blocs = append(blocs, Bloc(blockName,
				Valu(currVal, OpAdd64, intType, 0, nil, prevVal, "one"),
				Valu("i_inc", OpAdd64, intType, 0, nil, "i", "one"),
				Goto("header")))
		} else {
			blocs = append(blocs, Bloc(blockName,
				Valu(currVal, OpAdd64, intType, 0, nil, prevVal, "one"),
				Goto(nextBlock)))
		}
		prevVal = currVal
	}

	// Exit
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// =============================================================================
// NESTED LOOP CFG BUILDERS
// =============================================================================

// buildNestedLoops creates a CFG with N levels of nested loops:
//
//	entry
//	  │
//	  ▼
//	L1_header ◄─────────┐
//	  │ (cond)          │
//	  ├──► exit         │
//	  ▼                 │
//	L2_header ◄───────┐ │
//	  │ (cond)        │ │
//	  ├──► L1_latch ──┘ │
//	  ▼               │ │
//	 ...              │ │
//	  ▼               │ │
//	LN_header ◄─────┐ │ │
//	  │ (cond)      │ │ │
//	  ├──► L(N-1)_latch
//	  ▼             │
//	body            │
//	  │             │
//	  ▼             │
//	LN_latch ───────┘
//
// This creates realistic nested loop liveness patterns where:
// - Outer loop induction variables live across inner loops
// - Phi nodes at each header
// - Multiple back edges at different nesting levels
func buildNestedLoops(tb testing.TB, depth int) *Func {
	if depth < 1 {
		depth = 1
	}

	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	// Total blocks: entry + depth*(header + latch) + body + exit
	// = 2*depth + 3
	blocs := make([]bloc, 0, 2*depth+3)

	// Entry block: initialize constants and all loop counters' initial values
	entryVals := []any{
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("limit", OpConst64, intType, 10, nil),
	}
	// Initial values for each loop's induction variable
	for i := 1; i <= depth; i++ {
		initVal := fmt.Sprintf("init%d", i)
		entryVals = append(entryVals, Valu(initVal, OpConst64, intType, 0, nil))
	}
	entryVals = append(entryVals, Goto("L1_header"))
	blocs = append(blocs, Bloc("entry", entryVals...))

	// Create header and latch blocks for each nesting level
	for i := 1; i <= depth; i++ {
		headerName := fmt.Sprintf("L%d_header", i)
		latchName := fmt.Sprintf("L%d_latch", i)
		phiVar := fmt.Sprintf("i%d", i)
		incVar := fmt.Sprintf("i%d_inc", i)
		cmpVar := fmt.Sprintf("cmp%d", i)
		initVar := fmt.Sprintf("init%d", i)

		// Where does the header branch to?
		var trueTarget, falseTarget string
		if i < depth {
			// Enter next inner loop
			trueTarget = fmt.Sprintf("L%d_header", i+1)
		} else {
			// Innermost: enter body
			trueTarget = "body"
		}
		if i == 1 {
			// Outermost: exit the function
			falseTarget = "exit"
		} else {
			// Exit to outer loop's latch
			falseTarget = fmt.Sprintf("L%d_latch", i-1)
		}

		// Header block: phi for induction var, compare, branch
		blocs = append(blocs, Bloc(headerName,
			Valu(phiVar, OpPhi, intType, 0, nil, initVar, incVar),
			Valu(cmpVar, OpLess64, boolType, 0, nil, phiVar, "limit"),
			If(cmpVar, trueTarget, falseTarget)))

		// Latch block: increment, back edge to header
		blocs = append(blocs, Bloc(latchName,
			Valu(incVar, OpAdd64, intType, 0, nil, phiVar, "one"),
			Goto(headerName)))
	}

	// Body block: innermost loop body
	// Create some computation using all induction variables (creates cross-loop liveness)
	bodyVals := make([]any, 0, depth+2)

	// Accumulate all induction variables: sum = i1 + i2 + ... + iN
	prevSum := "i1"
	for i := 2; i <= depth; i++ {
		sumVar := fmt.Sprintf("sum%d", i)
		iVar := fmt.Sprintf("i%d", i)
		bodyVals = append(bodyVals, Valu(sumVar, OpAdd64, intType, 0, nil, prevSum, iVar))
		prevSum = sumVar
	}

	// Add a final computation to use the sum
	if depth > 1 {
		bodyVals = append(bodyVals, Valu("result", OpAdd64, intType, 0, nil, prevSum, "one"))
	} else {
		bodyVals = append(bodyVals, Valu("result", OpAdd64, intType, 0, nil, "i1", "one"))
	}

	// Jump to innermost latch
	innermostLatch := fmt.Sprintf("L%d_latch", depth)
	bodyVals = append(bodyVals, Goto(innermostLatch))
	blocs = append(blocs, Bloc("body", bodyVals...))

	// Exit block
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// buildNestedLoopsWithWork adds more computation per loop level
// to create denser liveness patterns
func buildNestedLoopsWithWork(tb testing.TB, depth, workPerLevel int) *Func {
	if depth < 1 {
		depth = 1
	}
	if workPerLevel < 1 {
		workPerLevel = 1
	}

	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	blocs := make([]bloc, 0, 2*depth+3)

	// Entry block
	entryVals := []any{
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("limit", OpConst64, intType, 10, nil),
		Valu("acc_init", OpConst64, intType, 0, nil), // Accumulator
	}
	for i := 1; i <= depth; i++ {
		initVal := fmt.Sprintf("init%d", i)
		entryVals = append(entryVals, Valu(initVal, OpConst64, intType, 0, nil))
	}
	entryVals = append(entryVals, Goto("L1_header"))
	blocs = append(blocs, Bloc("entry", entryVals...))

	// Track accumulator phi args for each level
	accPhiArgs := make([]string, depth+1)
	accPhiArgs[0] = "acc_init" // Entry provides initial value

	// Create header and latch blocks for each nesting level
	for i := 1; i <= depth; i++ {
		headerName := fmt.Sprintf("L%d_header", i)
		latchName := fmt.Sprintf("L%d_latch", i)
		phiVar := fmt.Sprintf("i%d", i)
		incVar := fmt.Sprintf("i%d_inc", i)
		cmpVar := fmt.Sprintf("cmp%d", i)
		initVar := fmt.Sprintf("init%d", i)
		accPhi := fmt.Sprintf("acc%d", i)

		// Determine branch targets
		var trueTarget, falseTarget string
		if i < depth {
			trueTarget = fmt.Sprintf("L%d_header", i+1)
		} else {
			trueTarget = "body"
		}
		if i == 1 {
			falseTarget = "exit"
		} else {
			falseTarget = fmt.Sprintf("L%d_latch", i-1)
		}

		// Determine accumulator phi incoming values
		var accIncoming1, accIncoming2 string
		if i == 1 {
			accIncoming1 = "acc_init"
		} else {
			accIncoming1 = fmt.Sprintf("acc%d", i-1) // From outer header's phi
		}
		accIncoming2 = fmt.Sprintf("acc%d_out", i) // From this level's latch

		// Header: phi for index, phi for accumulator, work, compare, branch
		headerVals := []any{
			Valu(phiVar, OpPhi, intType, 0, nil, initVar, incVar),
			Valu(accPhi, OpPhi, intType, 0, nil, accIncoming1, accIncoming2),
		}

		// Add work at this level using the accumulator and index
		prevVal := accPhi
		for w := 0; w < workPerLevel; w++ {
			workVar := fmt.Sprintf("work%d_%d", i, w)
			headerVals = append(headerVals,
				Valu(workVar, OpAdd64, intType, 0, nil, prevVal, phiVar))
			prevVal = workVar
		}
		workOutVar := fmt.Sprintf("work%d_out", i)
		headerVals = append(headerVals,
			Valu(workOutVar, OpAdd64, intType, 0, nil, prevVal, "one"))

		headerVals = append(headerVals,
			Valu(cmpVar, OpLess64, boolType, 0, nil, phiVar, "limit"),
			If(cmpVar, trueTarget, falseTarget))

		blocs = append(blocs, Bloc(headerName, headerVals...))

		// Latch: increment index, pass through accumulator from inner level
		var accFromInner string
		if i == depth {
			accFromInner = "body_result" // Innermost gets result from body
		} else {
			accFromInner = fmt.Sprintf("work%d_out", i+1) // From inner header
		}
		accOutVar := fmt.Sprintf("acc%d_out", i)

		blocs = append(blocs, Bloc(latchName,
			Valu(incVar, OpAdd64, intType, 0, nil, phiVar, "one"),
			Valu(accOutVar, OpCopy, intType, 0, nil, accFromInner),
			Goto(headerName)))
	}

	// Body block
	innermostAcc := fmt.Sprintf("acc%d", depth)
	innermostIdx := fmt.Sprintf("i%d", depth)
	innermostLatch := fmt.Sprintf("L%d_latch", depth)

	bodyVals := []any{
		Valu("body_tmp", OpAdd64, intType, 0, nil, innermostAcc, innermostIdx),
		Valu("body_result", OpAdd64, intType, 0, nil, "body_tmp", "one"),
		Goto(innermostLatch),
	}
	blocs = append(blocs, Bloc("body", bodyVals...))

	// Exit block
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// =============================================================================
// IRREDUCIBLE CFG BUILDERS
// =============================================================================

// buildIrreducibleSimple creates the classic irreducible CFG:
//
//	 entry
//	 /   \
//	▼     ▼
//	B ◄──► C    (mutual edges - irreducible!)
//	 \   /
//	  ▼ ▼
//	  exit
//
// B and C form an irreducible cycle with TWO entry points.
// This is the minimal irreducible CFG.
func buildIrreducibleSimple(tb testing.TB) *Func {
	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("v0", OpConst64, intType, 0, nil),
			Valu("v1", OpConst64, intType, 1, nil),
			Valu("cond_entry", OpConstBool, boolType, 1, nil),
			If("cond_entry", "B", "C")), // Two entries into the cycle!

		Bloc("B",
			Valu("phi_b", OpPhi, intType, 0, nil, "v0", "vc"), // From entry, from C
			Valu("vb", OpAdd64, intType, 0, nil, "phi_b", "v1"),
			Valu("cond_b", OpLess64, boolType, 0, nil, "vb", "v1"),
			If("cond_b", "C", "exit")), // Can go to C or exit

		Bloc("C",
			Valu("phi_c", OpPhi, intType, 0, nil, "v1", "vb"), // From entry, from B
			Valu("vc", OpAdd64, intType, 0, nil, "phi_c", "v1"),
			Valu("cond_c", OpLess64, boolType, 0, nil, "vc", "v1"),
			If("cond_c", "B", "exit")), // Can go to B or exit

		Bloc("exit",
			Exit("mem")))

	return fun.f
}

// buildIrreducibleDiamond creates an irreducible diamond pattern:
//
//	  entry
//	  /   \
//	 ▼     ▼
//	L1     R1
//	 |\   /|
//	 | \ / |
//	 |  X  |      <- Cross edges create irreducibility
//	 | / \ |
//	 ▼/   \▼
//	L2     R2
//	 \     /
//	  ▼   ▼
//	  exit
//
// The cross edges (L1→R2, R1→L2) create two irreducible regions.
func buildIrreducibleDiamond(tb testing.TB) *Func {
	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("v0", OpConst64, intType, 0, nil),
			Valu("cond0", OpConstBool, boolType, 1, nil),
			If("cond0", "L1", "R1")),

		Bloc("L1",
			Valu("vl1", OpAdd64, intType, 0, nil, "v0", "v0"),
			Valu("cond_l1", OpConstBool, boolType, 1, nil),
			If("cond_l1", "L2", "R2")), // L1 can go to BOTH L2 and R2

		Bloc("R1",
			Valu("vr1", OpAdd64, intType, 0, nil, "v0", "v0"),
			Valu("cond_r1", OpConstBool, boolType, 1, nil),
			If("cond_r1", "R2", "L2")), // R1 can go to BOTH R2 and L2

		Bloc("L2",
			Valu("phi_l2", OpPhi, intType, 0, nil, "vl1", "vr1"), // From L1, R1
			Valu("vl2", OpAdd64, intType, 0, nil, "phi_l2", "v0"),
			Goto("exit")),

		Bloc("R2",
			Valu("phi_r2", OpPhi, intType, 0, nil, "vl1", "vr1"), // From L1, R1
			Valu("vr2", OpAdd64, intType, 0, nil, "phi_r2", "v0"),
			Goto("exit")),

		Bloc("exit",
			Exit("mem")))

	return fun.f
}

// buildIrreducibleLoop creates an irreducible loop with N nodes:
//
//	      entry
//	      /   \
//	     ▼     ▼
//	┌─► N1 ──► N2 ──► N3 ──► ... ──► Nn ─┐
//	│    ▲                            │   │
//	│    └────────────────────────────┘   │
//	│                                     ▼
//	└─────────────────────────────────  exit
//
// Entry branches to BOTH N1 and N2, creating irreducibility.
// The cycle N1→N2→...→Nn→N1 has two entry points.
func buildIrreducibleLoop(tb testing.TB, n int) *Func {
	if n < 2 {
		n = 2
	}

	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	blocs := make([]bloc, 0, n+2)

	// Entry: branches to N1 and N2 (two entries into the cycle)
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("limit", OpConst64, intType, 100, nil),
		Valu("cond_entry", OpConstBool, boolType, 1, nil),
		If("cond_entry", "N1", "N2"))) // TWO entries into cycle!

	// Build cycle nodes N1, N2, ..., Nn
	for i := 1; i <= n; i++ {
		nodeName := fmt.Sprintf("N%d", i)
		phiVar := fmt.Sprintf("phi%d", i)
		valVar := fmt.Sprintf("v%d", i)
		cmpVar := fmt.Sprintf("cmp%d", i)

		// Determine phi incoming edges
		var phiArg1, phiArg2 string
		if i == 1 {
			// N1: from entry and from Nn (back edge)
			phiArg1 = "zero"
			phiArg2 = fmt.Sprintf("v%d", n)
		} else if i == 2 {
			// N2: from entry and from N1
			phiArg1 = "one" // Different initial value from entry
			phiArg2 = "v1"
		} else {
			// Ni: from N(i-1) only (but we need phi for SSA form)
			// Actually, let's make it simpler: only N1 and N2 have phis
			phiArg1 = fmt.Sprintf("v%d", i-1)
			phiArg2 = fmt.Sprintf("v%d", i-1)
		}

		// Determine successor
		var nextNode string
		if i == n {
			nextNode = "N1" // Back edge
		} else {
			nextNode = fmt.Sprintf("N%d", i+1)
		}

		// N1 and N2 have phi nodes due to multiple predecessors
		if i <= 2 {
			blocs = append(blocs, Bloc(nodeName,
				Valu(phiVar, OpPhi, intType, 0, nil, phiArg1, phiArg2),
				Valu(valVar, OpAdd64, intType, 0, nil, phiVar, "one"),
				Valu(cmpVar, OpLess64, boolType, 0, nil, valVar, "limit"),
				If(cmpVar, nextNode, "exit")))
		} else {
			// Other nodes: simple pass-through
			prevVal := fmt.Sprintf("v%d", i-1)
			blocs = append(blocs, Bloc(nodeName,
				Valu(valVar, OpAdd64, intType, 0, nil, prevVal, "one"),
				Valu(cmpVar, OpLess64, boolType, 0, nil, valVar, "limit"),
				If(cmpVar, nextNode, "exit")))
		}
	}

	// Exit
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// buildIrreducibleMultiEntry creates an irreducible region with M entry points:
//
//	       entry
//	  /   /  |  \   \
//	 ▼   ▼   ▼   ▼   ▼
//	N1  N2  N3  N4  N5  ...  (M entries)
//	 \   \   |   /   /
//	  ▼   ▼  ▼  ▼   ▼
//	     merge
//	       │
//	       ▼ (back to entry for a loop)
//	     exit
//
// Entry fans out to M nodes, all of which merge back.
// Adding a back edge from merge to any Ni creates irreducibility.
func buildIrreducibleMultiEntry(tb testing.TB, entries int) *Func {
	if entries < 2 {
		entries = 2
	}

	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	blocs := make([]bloc, 0, entries+3)

	// Entry block with a chain of conditionals to reach each Ni
	// We'll use a switch-like pattern
	entryVals := []any{
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
	}

	// For simplicity, entry goes to dispatch block
	entryVals = append(entryVals, Goto("dispatch"))
	blocs = append(blocs, Bloc("entry", entryVals...))

	// Dispatch block: creates multiple branches
	// We'll chain If statements through intermediate blocks
	for i := 1; i < entries; i++ {
		dispatchName := fmt.Sprintf("dispatch%d", i)
		condVar := fmt.Sprintf("dcond%d", i)
		targetNode := fmt.Sprintf("N%d", i)

		var nextDispatch string
		if i == entries-1 {
			nextDispatch = fmt.Sprintf("N%d", entries) // Last one goes to final N
		} else {
			nextDispatch = fmt.Sprintf("dispatch%d", i+1)
		}

		if i == 1 {
			// First dispatch
			blocs = append(blocs, Bloc("dispatch",
				Valu(condVar, OpConstBool, boolType, 1, nil),
				If(condVar, targetNode, nextDispatch)))
		} else {
			blocs = append(blocs, Bloc(dispatchName,
				Valu(condVar, OpConstBool, boolType, 1, nil),
				If(condVar, targetNode, nextDispatch)))
		}
	}

	// Create N1, N2, ..., N_entries nodes
	// Each Ni can receive from dispatch AND from merge (back edge) - irreducible!
	for i := 1; i <= entries; i++ {
		nodeName := fmt.Sprintf("N%d", i)
		valVar := fmt.Sprintf("vn%d", i)
		phiVar := fmt.Sprintf("phi_n%d", i)

		// Phi: from dispatch and from merge (back edge)
		blocs = append(blocs, Bloc(nodeName,
			Valu(phiVar, OpPhi, intType, 0, nil, "zero", fmt.Sprintf("vm%d", i)),
			Valu(valVar, OpAdd64, intType, 0, nil, phiVar, "one"),
			Goto("merge")))
	}

	// Merge block: collects all Ni, can branch back to any Ni (irreducible!)
	mergeVals := []any{}

	// Phi to merge all incoming values
	phiArgs := make([]string, entries)
	for i := 1; i <= entries; i++ {
		phiArgs[i-1] = fmt.Sprintf("vn%d", i)
	}
	mergeVals = append(mergeVals, Valu("phi_merge", OpPhi, intType, 0, nil, phiArgs...))
	mergeVals = append(mergeVals, Valu("vm_sum", OpAdd64, intType, 0, nil, "phi_merge", "one"))

	// Create values that will be used by the back-edge phis
	for i := 1; i <= entries; i++ {
		vmVar := fmt.Sprintf("vm%d", i)
		mergeVals = append(mergeVals, Valu(vmVar, OpAdd64, intType, 0, nil, "vm_sum", "one"))
	}

	// Exit condition and back edge to N1 (creates the irreducible cycle)
	mergeVals = append(mergeVals,
		Valu("cmp_merge", OpLess64, boolType, 0, nil, "vm_sum", "one"),
		If("cmp_merge", "N1", "exit")) // Back edge to N1!

	blocs = append(blocs, Bloc("merge", mergeVals...))

	// Exit
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// buildIrreducibleNested creates nested irreducible regions:
//
//	    entry
//	    /   \
//	   ▼     ▼
//	  A1 ◄──► A2     (outer irreducible)
//	   |
//	   ▼
//	  /   \
//	 ▼     ▼
//	B1 ◄──► B2       (inner irreducible)
//	 \     /
//	  ▼   ▼
//	  exit
func buildIrreducibleNested(tb testing.TB, depth int) *Func {
	if depth < 1 {
		depth = 1
	}
	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool
	blocs := make([]bloc, 0, depth*2+2)

	// Entry
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("cond_entry", OpConstBool, boolType, 1, nil),
		If("cond_entry", "L1_A", "L1_B"))) // First irreducible region

	// Build nested irreducible regions
	for level := 1; level <= depth; level++ {
		nodeA := fmt.Sprintf("L%d_A", level)
		nodeB := fmt.Sprintf("L%d_B", level)
		phiA := fmt.Sprintf("phi_a%d", level)
		phiB := fmt.Sprintf("phi_b%d", level)
		valA := fmt.Sprintf("va%d", level)
		valB := fmt.Sprintf("vb%d", level)
		condA := fmt.Sprintf("cond_a%d", level)
		condB := fmt.Sprintf("cond_b%d", level)

		// Determine where A and B can go
		var nextLevelA, nextLevelB string
		if level < depth {
			// Go to next level's irreducible region
			nextLevelA = fmt.Sprintf("L%d_A", level+1)
			nextLevelB = fmt.Sprintf("L%d_B", level+1)
		} else {
			// Innermost: go to exit
			nextLevelA = "exit"
			nextLevelB = "exit"
		}
		// Removed: exitTarget is no longer needed

		// Determine phi inputs
		var phiArgA1, phiArgA2, phiArgB1, phiArgB2 string
		if level == 1 {
			phiArgA1 = "zero" // From entry
			phiArgB1 = "one"  // From entry
		} else {
			// From previous level
			phiArgA1 = fmt.Sprintf("va%d", level-1)
			phiArgB1 = fmt.Sprintf("vb%d", level-1)
		}
		phiArgA2 = valB // From B at same level (cross edge)
		phiArgB2 = valA // From A at same level (cross edge)

		// Node A: part of irreducible pair with B
		blocs = append(blocs, Bloc(nodeA,
			Valu(phiA, OpPhi, intType, 0, nil, phiArgA1, phiArgA2),
			Valu(valA, OpAdd64, intType, 0, nil, phiA, "one"),
			Valu(condA, OpConstBool, boolType, 1, nil),
			If(condA, nodeB, nextLevelA))) // Can go to B (cross) or next level

		// Node B: part of irreducible pair with A
		blocs = append(blocs, Bloc(nodeB,
			Valu(phiB, OpPhi, intType, 0, nil, phiArgB1, phiArgB2),
			Valu(valB, OpAdd64, intType, 0, nil, phiB, "one"),
			Valu(condB, OpConstBool, boolType, 1, nil),
			If(condB, nodeA, nextLevelB))) // ← Fixed: use nextLevelB
	}

	// Exit
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// buildIrreducibleWithReducible creates a mix of reducible and irreducible regions:
//
//	  entry
//	    │
//	    ▼
//	┌─► header ◄─┐     (reducible loop)
//	│     │      │
//	│     ▼      │
//	│   /   \    │
//	│  ▼     ▼   │
//	│ B1 ◄──► B2 │     (irreducible region INSIDE the loop)
//	│  \     /   │
//	│   ▼   ▼    │
//	│   merge    │
//	│     │      │
//	│     ├──────┘     (back edge - reducible)
//	│     ▼
//	│   exit
func buildIrreducibleWithReducible(tb testing.TB, loopSize int) *Func {
	if loopSize < 2 {
		loopSize = 2
	}

	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool

	blocs := make([]bloc, 0, loopSize+5)

	// Entry
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("limit", OpConst64, intType, 100, nil),
		Goto("header")))

	// Loop header (reducible)
	blocs = append(blocs, Bloc("header",
		Valu("i", OpPhi, intType, 0, nil, "zero", "i_inc"),
		Valu("cmp_header", OpLess64, boolType, 0, nil, "i", "limit"),
		If("cmp_header", "dispatch", "exit")))

	// Dispatch into irreducible region
	blocs = append(blocs, Bloc("dispatch",
		Valu("cond_dispatch", OpConstBool, boolType, 1, nil),
		If("cond_dispatch", "B1", "B2"))) // TWO entries!

	// B1 and B2 form irreducible region
	blocs = append(blocs, Bloc("B1",
		Valu("phi_b1", OpPhi, intType, 0, nil, "i", "vb2"), // From dispatch, from B2
		Valu("vb1", OpAdd64, intType, 0, nil, "phi_b1", "one"),
		Valu("cond_b1", OpConstBool, boolType, 1, nil),
		If("cond_b1", "B2", "merge")))

	blocs = append(blocs, Bloc("B2",
		Valu("phi_b2", OpPhi, intType, 0, nil, "i", "vb1"), // From dispatch, from B1
		Valu("vb2", OpAdd64, intType, 0, nil, "phi_b2", "one"),
		Valu("cond_b2", OpConstBool, boolType, 1, nil),
		If("cond_b2", "B1", "merge")))

	// Merge and loop back
	blocs = append(blocs, Bloc("merge",
		Valu("phi_merge", OpPhi, intType, 0, nil, "vb1", "vb2"),
		Valu("i_inc", OpAdd64, intType, 0, nil, "i", "one"),
		Goto("header"))) // Back edge (reducible)

	// Exit
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// buildIrreducibleChain creates a chain of irreducible regions:
//
//	entry → [irreducible1] → [irreducible2] → ... → [irreducibleN] → exit
//
// Each irreducible region is the simple B↔C pattern.
func buildIrreducibleChain(tb testing.TB, numRegions int) *Func {
	if numRegions < 1 {
		numRegions = 1
	}
	c := testConfig(tb)
	intType := c.config.Types.Int64
	boolType := c.config.Types.Bool
	blocs := make([]bloc, 0, numRegions*2+2) // ← Also fixed: *2 not *3 (no merge nodes)

	// Entry
	blocs = append(blocs, Bloc("entry",
		Valu("mem", OpInitMem, types.TypeMem, 0, nil),
		Valu("zero", OpConst64, intType, 0, nil),
		Valu("one", OpConst64, intType, 1, nil),
		Valu("cond_entry", OpConstBool, boolType, 1, nil),
		If("cond_entry", "R1_A", "R1_B")))

	// Build chain of irreducible regions
	for r := 1; r <= numRegions; r++ {
		nodeA := fmt.Sprintf("R%d_A", r)
		nodeB := fmt.Sprintf("R%d_B", r)
		// Removed: mergeNode (unused)

		phiA := fmt.Sprintf("phi_a%d", r)
		phiB := fmt.Sprintf("phi_b%d", r)
		valA := fmt.Sprintf("va%d", r)
		valB := fmt.Sprintf("vb%d", r)
		condA := fmt.Sprintf("cond_a%d", r)
		condB := fmt.Sprintf("cond_b%d", r)

		// Determine next region or exit
		var nextRegionA, nextRegionB string
		if r < numRegions {
			nextRegionA = fmt.Sprintf("R%d_A", r+1)
			nextRegionB = fmt.Sprintf("R%d_B", r+1)
		} else {
			nextRegionA = "exit"
			nextRegionB = "exit"
		}

		// Phi inputs from previous region or entry
		var prevValA, prevValB string
		if r == 1 {
			prevValA = "zero"
			prevValB = "one"
		} else {
			prevValA = fmt.Sprintf("va%d", r-1)
			prevValB = fmt.Sprintf("vb%d", r-1)
		}

		// Node A
		blocs = append(blocs, Bloc(nodeA,
			Valu(phiA, OpPhi, intType, 0, nil, prevValA, valB),
			Valu(valA, OpAdd64, intType, 0, nil, phiA, "one"),
			Valu(condA, OpConstBool, boolType, 1, nil),
			If(condA, nodeB, nextRegionA)))

		// Node B
		blocs = append(blocs, Bloc(nodeB,
			Valu(phiB, OpPhi, intType, 0, nil, prevValB, valA),
			Valu(valB, OpAdd64, intType, 0, nil, phiB, "one"),
			Valu(condB, OpConstBool, boolType, 1, nil),
			If(condB, nodeA, nextRegionB)))
	}

	// Exit
	blocs = append(blocs, Bloc("exit",
		Exit("mem")))

	fun := c.Fun("entry", blocs...)
	return fun.f
}

// =============================================================================
// REAL LIFE CFG EXAMPLES
// =============================================================================

// buildGoldenSHA1TestCFG builds the CFG from TestGolden in crypto/sha1
// This has:
// - Outer loop (b2 -> b4 -> b2) iterating over golden test cases
// - Inner loop (b17 -> b19 -> b17) for j=0..3
// - 5 type-assertion mini-loops
// - Multiple conditional branches and phi nodes
func buildGoldenSHA1TestCFG(c *Conf) *Func {
	ptrType := c.config.Types.BytePtr
	intType := c.config.Types.Int
	memType := types.TypeMem
	boolType := c.config.Types.Bool

	fun := c.Fun("b1",
		// Entry block
		Bloc("b1",
			Valu("mem", OpInitMem, memType, 0, nil),
			Valu("sp", OpSP, ptrType, 0, nil),
			Valu("sb", OpSB, ptrType, 0, nil),
			Valu("zero_int", OpConst64, intType, 0, nil),
			Valu("zero_ptr", OpConstNil, ptrType, 0, nil),
			Goto("b2")),

		// Main loop header - iterates over golden test cases
		Bloc("b2",
			Valu("i", OpPhi, intType, 0, nil, "zero_int", "i_inc"),
			Valu("mem2", OpPhi, memType, 0, nil, "mem", "mem930"),
			Valu("cmp_i", OpLess64, boolType, 0, nil, "i", "golden_len"),
			Valu("golden_len", OpConst64, intType, 10, nil), // simulated length
			If("cmp_i", "b8", "b5")),

		// Process test case - check sum result
		Bloc("b8",
			Valu("cmp_ne", OpNeq64, boolType, 0, nil, "i", "zero_int"),
			If("cmp_ne", "b105", "b12")),

		Bloc("b105",
			Goto("b13")),

		Bloc("b12",
			Valu("mem_eq", OpLoad, boolType, 0, nil, "sp", "mem2"),
			If("mem_eq", "b86", "b90")),

		Bloc("b86",
			Goto("b16")),

		Bloc("b90",
			Goto("b13")),

		// Error path - call Fatalf
		Bloc("b13",
			Valu("mem13", OpStaticCall, memType, 0, nil, "mem2"),
			Goto("b16")),

		// Setup digest and enter inner loop
		Bloc("b16",
			Valu("mem16", OpPhi, memType, 0, nil, "mem2", "mem13"),
			Valu("digest", OpStaticCall, ptrType, 0, nil, "mem16"),
			Valu("mem16b", OpStaticCall, memType, 0, nil, "mem16"),
			Goto("b17")),

		// Inner loop header - j = 0..3
		Bloc("b17",
			Valu("j", OpPhi, intType, 0, nil, "zero_int", "j_inc"),
			Valu("mem17", OpPhi, memType, 0, nil, "mem16b", "mem875"),
			Valu("four", OpConst64, intType, 4, nil),
			Valu("cmp_j", OpLess64, boolType, 0, nil, "j", "four"),
			If("cmp_j", "b25", "b4")),

		// Branch on j value
		Bloc("b25",
			Valu("one", OpConst64, intType, 1, nil),
			Valu("cmp_j_le1", OpLeq64, boolType, 0, nil, "j", "one"),
			If("cmp_j_le1", "b35", "b24")),

		// j <= 1: typeAssert.0 path
		Bloc("b35",
			Valu("type_ptr0", OpLoad, ptrType, 0, nil, "sb", "mem17"),
			Goto("b37")),

		// j > 1: check j == 2
		Bloc("b24",
			Valu("two", OpConst64, intType, 2, nil),
			Valu("cmp_j_eq2", OpEq64, boolType, 0, nil, "j", "two"),
			If("cmp_j_eq2", "b48", "b29")),

		// j == 2: typeAssert.1 path
		Bloc("b48",
			Goto("b50")),

		// j > 2: check j == 3
		Bloc("b29",
			Valu("three", OpConst64, intType, 3, nil),
			Valu("cmp_j_eq3", OpEq64, boolType, 0, nil, "j", "three"),
			If("cmp_j_eq3", "b72", "b22")),

		// j == 3: typeAssert.3 path
		Bloc("b72",
			Goto("b74")),

		// j > 3 (shouldn't happen): direct to b103
		Bloc("b22",
			Goto("b103")),

		// Back edge for outer loop
		Bloc("b4",
			Valu("i_inc", OpAdd64, intType, 0, nil, "i", "one"),
			Goto("b2")),

		// Type assertion loop 0: b37 -> b38 -> b73 -> b37
		Bloc("b37",
			Valu("ta0_ptr", OpPhi, ptrType, 0, nil, "type_ptr0", "ta0_ptr_inc"),
			Valu("ta0_found", OpEq64, boolType, 0, nil, "ta0_ptr", "zero_ptr"),
			If("ta0_found", "b39", "b38")),

		Bloc("b38",
			Valu("ta0_ptr_inc", OpAdd64, ptrType, 0, nil, "ta0_ptr", "one"),
			Valu("ta0_nz", OpNeq64, boolType, 0, nil, "ta0_ptr", "zero_ptr"),
			If("ta0_nz", "b73", "b40")),

		Bloc("b73",
			Goto("b37")),

		Bloc("b40",
			Valu("mem40", OpStaticCall, memType, 0, nil, "mem17"),
			Goto("b41")),

		Bloc("b39",
			Valu("ta0_result", OpLoad, ptrType, 0, nil, "ta0_ptr", "mem17"),
			Goto("b41")),

		Bloc("b41",
			Valu("mem41", OpPhi, memType, 0, nil, "mem40", "mem17"),
			Valu("ta0_final", OpPhi, ptrType, 0, nil, "ta0_result", "ta0_result"),
			Valu("mem41b", OpStaticCall, memType, 0, nil, "mem41"),
			Goto("b103")),

		// Type assertion loop 1: b50 -> b51 -> b69 -> b50
		Bloc("b50",
			Valu("ta1_ptr", OpPhi, ptrType, 0, nil, "zero_ptr", "ta1_ptr_inc"),
			Valu("ta1_found", OpEq64, boolType, 0, nil, "ta1_ptr", "zero_ptr"),
			If("ta1_found", "b52", "b51")),

		Bloc("b51",
			Valu("ta1_ptr_inc", OpAdd64, ptrType, 0, nil, "ta1_ptr", "one"),
			Valu("ta1_nz", OpNeq64, boolType, 0, nil, "ta1_ptr", "zero_ptr"),
			If("ta1_nz", "b69", "b53")),

		Bloc("b69",
			Goto("b50")),

		Bloc("b53",
			Valu("mem53", OpStaticCall, memType, 0, nil, "mem17"),
			Goto("b59")),

		Bloc("b52",
			Valu("ta1_result", OpLoad, ptrType, 0, nil, "ta1_ptr", "mem17"),
			Goto("b59")),

		Bloc("b59",
			Valu("mem59", OpPhi, memType, 0, nil, "mem53", "mem17"),
			Valu("mem59b", OpStaticCall, memType, 0, nil, "mem59"),
			Goto("b61")),

		// Type assertion loop 2: b61 -> b62 -> b60 -> b61
		Bloc("b61",
			Valu("ta2_ptr", OpPhi, ptrType, 0, nil, "zero_ptr", "ta2_ptr_inc"),
			Valu("ta2_found", OpEq64, boolType, 0, nil, "ta2_ptr", "zero_ptr"),
			If("ta2_found", "b63", "b62")),

		Bloc("b62",
			Valu("ta2_ptr_inc", OpAdd64, ptrType, 0, nil, "ta2_ptr", "one"),
			Valu("ta2_nz", OpNeq64, boolType, 0, nil, "ta2_ptr", "zero_ptr"),
			If("ta2_nz", "b60", "b64")),

		Bloc("b60",
			Goto("b61")),

		Bloc("b64",
			Valu("mem64", OpStaticCall, memType, 0, nil, "mem59b"),
			Goto("b65")),

		Bloc("b63",
			Valu("ta2_result", OpLoad, ptrType, 0, nil, "ta2_ptr", "mem59b"),
			Goto("b65")),

		Bloc("b65",
			Valu("mem65", OpPhi, memType, 0, nil, "mem64", "mem59b"),
			Valu("mem65b", OpStaticCall, memType, 0, nil, "mem65"),
			Goto("b103")),

		// Type assertion loop 3: b74 -> b75 -> b56 -> b74
		Bloc("b74",
			Valu("ta3_ptr", OpPhi, ptrType, 0, nil, "zero_ptr", "ta3_ptr_inc"),
			Valu("ta3_found", OpEq64, boolType, 0, nil, "ta3_ptr", "zero_ptr"),
			If("ta3_found", "b76", "b75")),

		Bloc("b75",
			Valu("ta3_ptr_inc", OpAdd64, ptrType, 0, nil, "ta3_ptr", "one"),
			Valu("ta3_nz", OpNeq64, boolType, 0, nil, "ta3_ptr", "zero_ptr"),
			If("ta3_nz", "b56", "b77")),

		Bloc("b56",
			Goto("b74")),

		Bloc("b77",
			Valu("mem77", OpStaticCall, memType, 0, nil, "mem17"),
			Goto("b89")),

		Bloc("b76",
			Valu("ta3_result", OpLoad, ptrType, 0, nil, "ta3_ptr", "mem17"),
			Goto("b89")),

		Bloc("b89",
			Valu("mem89", OpPhi, memType, 0, nil, "mem77", "mem17"),
			Valu("mem89b", OpStaticCall, memType, 0, nil, "mem89"),
			Goto("b91")),

		// Type assertion loop 4: b91 -> b92 -> b54 -> b91
		Bloc("b91",
			Valu("ta4_ptr", OpPhi, ptrType, 0, nil, "zero_ptr", "ta4_ptr_inc"),
			Valu("ta4_found", OpEq64, boolType, 0, nil, "ta4_ptr", "zero_ptr"),
			If("ta4_found", "b93", "b92")),

		Bloc("b92",
			Valu("ta4_ptr_inc", OpAdd64, ptrType, 0, nil, "ta4_ptr", "one"),
			Valu("ta4_nz", OpNeq64, boolType, 0, nil, "ta4_ptr", "zero_ptr"),
			If("ta4_nz", "b54", "b94")),

		Bloc("b54",
			Goto("b91")),

		Bloc("b94",
			Valu("mem94", OpStaticCall, memType, 0, nil, "mem89b"),
			Goto("b102")),

		Bloc("b93",
			Valu("ta4_result", OpLoad, ptrType, 0, nil, "ta4_ptr", "mem89b"),
			Goto("b102")),

		Bloc("b102",
			Valu("mem102", OpPhi, memType, 0, nil, "mem94", "mem89b"),
			Valu("mem102b", OpStaticCall, memType, 0, nil, "mem102"),
			Goto("b103")),

		// Merge point for all j paths - check result and loop back
		Bloc("b103",
			Valu("mem103", OpPhi, memType, 0, nil, "mem17", "mem41b", "mem65b", "mem102b"),
			Valu("sum_ptr", OpPhi, ptrType, 0, nil, "zero_ptr", "zero_ptr", "zero_ptr", "zero_ptr"),
			Valu("sum_len", OpPhi, intType, 0, nil, "zero_int", "zero_int", "zero_int", "zero_int"),
			Valu("cmp_sum", OpNeq64, boolType, 0, nil, "sum_len", "zero_int"),
			If("cmp_sum", "b49", "b107")),

		Bloc("b49",
			Goto("b108")),

		Bloc("b107",
			Valu("sum_eq", OpLoad, boolType, 0, nil, "sp", "mem103"),
			If("sum_eq", "b85", "b45")),

		Bloc("b85",
			Goto("b19")),

		Bloc("b45",
			Goto("b108")),

		// Error path for sum mismatch
		Bloc("b108",
			Valu("mem108", OpStaticCall, memType, 0, nil, "mem103"),
			Goto("b19")),

		// Reset digest and loop back
		Bloc("b19",
			Valu("mem875", OpPhi, memType, 0, nil, "mem103", "mem108"),
			Valu("j_inc", OpAdd64, intType, 0, nil, "j", "one"),
			Valu("mem930", OpStaticCall, memType, 0, nil, "mem875"),
			Goto("b17")),

		// Exit
		Bloc("b5",
			Exit("mem2")),
	)

	return fun.f
}

func TestGoldenSHA1CFGStructure(t *testing.T) {
	c := testConfig(t)
	f := buildGoldenSHA1TestCFG(c)

	// Count blocks we actually created
	expectedBlocks := 56 // Actual count from our Bloc() calls
	if len(f.Blocks) != expectedBlocks {
		t.Errorf("expected %d blocks, got %d", expectedBlocks, len(f.Blocks))
	}

	// Verify entry block is first block
	if f.Entry != f.Blocks[0] {
		t.Errorf("expected entry block to be first block")
	}

	// Count structural properties instead of checking specific names
	var (
		exitBlocks int
		ifBlocks   int
		gotoBlocks int
		phiCount   int
		backEdges  int
	)

	blockIndex := make(map[*Block]int)
	for i, b := range f.Blocks {
		blockIndex[b] = i
	}

	for _, b := range f.Blocks {
		// Count block types by successor count
		switch len(b.Succs) {
		case 0:
			exitBlocks++
		case 1:
			gotoBlocks++
		case 2:
			ifBlocks++
		}

		// Count phi nodes
		for _, v := range b.Values {
			if v.Op == OpPhi {
				phiCount++
			}
		}

		// Count back-edges (successor with lower index = back edge in RPO-ish order)
		for _, e := range b.Succs {
			succIdx := blockIndex[e.Block()]
			myIdx := blockIndex[b]
			if succIdx < myIdx {
				backEdges++
			}
		}
	}

	t.Logf("CFG structure:")
	t.Logf("  Total blocks: %d", len(f.Blocks))
	t.Logf("  Exit blocks: %d", exitBlocks)
	t.Logf("  If blocks (2 succs): %d", ifBlocks)
	t.Logf("  Goto blocks (1 succ): %d", gotoBlocks)
	t.Logf("  Phi nodes: %d", phiCount)
	t.Logf("  Back-edges: %d", backEdges)

	// Verify expected structural properties
	if exitBlocks != 1 {
		t.Errorf("expected 1 exit block, got %d", exitBlocks)
	}

	// We expect 7 back-edges:
	// - b4 -> b2 (outer loop)
	// - b19 -> b17 (inner loop)
	// - b73 -> b37, b69 -> b50, b60 -> b61, b56 -> b74, b54 -> b91 (5 type-assert loops)
	expectedBackEdges := 7
	if backEdges != expectedBackEdges {
		t.Errorf("expected %d back-edges, got %d", expectedBackEdges, backEdges)
	}

	// Verify we have a reasonable number of conditional branches
	// From the CFG: many If blocks for the conditionals
	if ifBlocks < 19 {
		t.Errorf("expected at least 19 if-blocks, got %d", ifBlocks)
	}
}

// =============================================================================
// ACYCLIC BENCHMARKS
// =============================================================================

func BenchmarkComputeLive_Acyclic_500(b *testing.B) {
	f := buildLinearChain(b, 500)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_AcyclicDense_100x5(b *testing.B) {
	f := buildLinearChainDense(b, 100, 5)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_AcyclicDense_200x20(b *testing.B) {
	f := buildLinearChainDense(b, 200, 20)
	benchmarkComputeLive(b, f)
}

// =============================================================================
// LOOPY BENCHMARKS
// =============================================================================

func BenchmarkComputeLive_Loop_10(b *testing.B) {
	f := buildSimpleLoop(b, 8)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Loop_100(b *testing.B) {
	f := buildSimpleLoop(b, 98)
	benchmarkComputeLive(b, f)
}

// Benchmarks for nested loops at various depths
func BenchmarkComputeLive_Nested_3(b *testing.B) {
	f := buildNestedLoops(b, 3)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Nested_5(b *testing.B) {
	f := buildNestedLoops(b, 5)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Nested_10(b *testing.B) {
	f := buildNestedLoops(b, 10)
	benchmarkComputeLive(b, f)
}

// Dense variants with more work per loop level
func BenchmarkComputeLive_NestedDense_3x5(b *testing.B) {
	f := buildNestedLoopsWithWork(b, 3, 5)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_NestedDense_5x3(b *testing.B) {
	f := buildNestedLoopsWithWork(b, 5, 3)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_NestedDense_5x20(b *testing.B) {
	f := buildNestedLoopsWithWork(b, 5, 20)
	benchmarkComputeLive(b, f)
}

// =============================================================================
// IRREDUCIBLE BENCHMARKS
// =============================================================================

// Simple irreducible (minimal case)
func BenchmarkComputeLive_Irreducible_Simple(b *testing.B) {
	f := buildIrreducibleSimple(b)
	benchmarkComputeLive(b, f)
}

// Diamond pattern
func BenchmarkComputeLive_Irreducible_Diamond(b *testing.B) {
	f := buildIrreducibleDiamond(b)
	benchmarkComputeLive(b, f)
}

// Irreducible loops of various sizes
func BenchmarkComputeLive_Irreducible_Loop5(b *testing.B) {
	f := buildIrreducibleLoop(b, 5)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Irreducible_Loop10(b *testing.B) {
	f := buildIrreducibleLoop(b, 10)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Irreducible_Loop20(b *testing.B) {
	f := buildIrreducibleLoop(b, 20)
	benchmarkComputeLive(b, f)
}

// Multi-entry irreducible regions
func BenchmarkComputeLive_Irreducible_MultiEntry5(b *testing.B) {
	f := buildIrreducibleMultiEntry(b, 5)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Irreducible_MultiEntry10(b *testing.B) {
	f := buildIrreducibleMultiEntry(b, 10)
	benchmarkComputeLive(b, f)
}

// Nested irreducible regions
func BenchmarkComputeLive_Irreducible_Nested5(b *testing.B) {
	f := buildIrreducibleNested(b, 5)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Irreducible_Nested10(b *testing.B) {
	f := buildIrreducibleNested(b, 10)
	benchmarkComputeLive(b, f)
}

// Mixed reducible + irreducible
func BenchmarkComputeLive_Irreducible_WithReducible(b *testing.B) {
	f := buildIrreducibleWithReducible(b, 5)
	benchmarkComputeLive(b, f)
}

// Chain of irreducible regions
func BenchmarkComputeLive_Irreducible_Chain3(b *testing.B) {
	f := buildIrreducibleChain(b, 3)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_Irreducible_Chain10(b *testing.B) {
	f := buildIrreducibleChain(b, 10)
	benchmarkComputeLive(b, f)
}

// =============================================================================
// REAL LIFE BENCHMARKS
// =============================================================================

func BenchmarkComputeLive_GoldenSHA1(b *testing.B) {
	c := testConfig(b)
	f := buildGoldenSHA1TestCFG(c)
	benchmarkComputeLive(b, f)
}

// Core benchmark runner
func benchmarkComputeLive(b *testing.B, f *Func) {
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		s := &regAllocState{}
		s.init(f)
		s.computeLive()
	}
}

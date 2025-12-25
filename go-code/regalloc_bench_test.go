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

// buildHeapSortCFG builds the CFG from sort.heapSort (pdqsort helper)
// This has:
// - Outer loop (b2 -> b4 -> b2) for j iterations
// - Inner loop (b6 -> b8 -> b6) for siftDown i iterations
// - Heapify down loop (b18 -> b20 -> b18)
// - Heapify up loop (b26 -> b28 -> b26)
// - Multiple conditional branches and early returns
func buildHeapSortCFG(c *Conf) *Func {
	ptrType := c.config.Types.BytePtr
	intType := c.config.Types.Int
	memType := types.TypeMem
	boolType := c.config.Types.Bool

	fun := c.Fun("b1",
		// Entry block - function arguments
		Bloc("b1",
			Valu("a", OpArg, intType, 0, nil),
			Valu("b", OpArg, intType, 0, nil),
			Valu("data_less", OpArg, ptrType, 0, nil),
			Valu("data_swap", OpArg, ptrType, 0, nil),
			Valu("mem", OpInitMem, memType, 0, nil),
			Valu("zero_bool", OpConstBool, boolType, 0, nil),
			Valu("zero_int", OpConst64, intType, 0, nil),
			Valu("one", OpConst64, intType, 1, nil),
			Valu("two", OpConst64, intType, 2, nil),
			Valu("five", OpConst64, intType, 5, nil),
			Valu("fifty", OpConst64, intType, 50, nil),
			Valu("i_init", OpAdd64, intType, 0, nil, "a", "one"),
			Goto("b2")),

		// Outer loop header - j < 5
		Bloc("b2",
			Valu("j", OpPhi, intType, 0, nil, "zero_int", "j_inc_outer"),
			Valu("mem2", OpPhi, memType, 0, nil, "mem", "mem4"),
			Valu("i_outer", OpPhi, intType, 0, nil, "i_init", "i6"),
			Valu("cmp_j_5", OpLess64, boolType, 0, nil, "j", "five"),
			If("cmp_j_5", "b3", "b5")),

		// Continue to inner loop
		Bloc("b3",
			Goto("b6")),

		// Exit - return false
		Bloc("b5",
			Valu("ret_false", OpConstBool, boolType, 0, nil),
			Exit("mem2")),

		// Inner loop header - siftDown: b > i
		Bloc("b6",
			Valu("i6", OpPhi, intType, 0, nil, "i_outer", "i_inc"),
			Valu("mem6", OpPhi, memType, 0, nil, "mem2", "mem10"),
			Valu("cmp_b_i", OpLess64, boolType, 0, nil, "i6", "b"),
			If("cmp_b_i", "b11", "b24")),

		// Fallthrough when b <= i
		Bloc("b24",
			Goto("b10")),

		// Call data.Less(i, i-1)
		Bloc("b11",
			Valu("i_minus_1", OpSub64, intType, 0, nil, "i6", "one"),
			Valu("less_ptr", OpLoad, ptrType, 0, nil, "data_less", "mem6"),
			Valu("call_less", OpClosureCall, types.TypeResultMem, 0, nil, "less_ptr", "data_less", "i6", "i_minus_1", "mem6"),
			Valu("less_result", OpSelectN, boolType, 0, nil, "call_less"),
			Valu("mem11", OpSelectN, memType, 1, nil, "call_less"),
			Valu("not_less", OpNot, boolType, 0, nil, "less_result"),
			Goto("b10")),

		// Merge and check result
		Bloc("b10",
			Valu("mem10", OpPhi, memType, 0, nil, "mem6", "mem11"),
			Valu("should_continue", OpPhi, boolType, 0, nil, "zero_bool", "not_less"),
			If("should_continue", "b8", "b9")),

		// Continue inner loop - increment i
		Bloc("b8",
			Valu("i_inc", OpAdd64, intType, 0, nil, "i6", "one"),
			Goto("b6")),

		// Exit inner loop - check if done
		Bloc("b9",
			Valu("cmp_eq", OpEq64, boolType, 0, nil, "b", "i6"),
			If("cmp_eq", "b13", "b12")),

		// Return true - found
		Bloc("b13",
			Valu("true_val", OpConstBool, boolType, 1, nil),
			Exit("mem10")),

		// Check threshold: b - a < 50
		Bloc("b12",
			Valu("diff", OpSub64, intType, 0, nil, "b", "a"),
			Valu("cmp_50", OpLess64, boolType, 0, nil, "diff", "fifty"),
			If("cmp_50", "b15", "b14")),

		// Return false - below threshold
		Bloc("b15",
			Exit("mem10")),

		// Above threshold - call swap and continue
		Bloc("b14",
			Valu("swap_ptr", OpLoad, ptrType, 0, nil, "data_swap", "mem10"),
			Valu("i_minus_1_b14", OpSub64, intType, 0, nil, "i6", "one"),
			Valu("call_swap", OpClosureCall, memType, 0, nil, "swap_ptr", "data_swap", "i6", "i_minus_1_b14", "mem10"),
			Valu("mem14", OpSelectN, memType, 0, nil, "call_swap"),
			Valu("diff2", OpSub64, intType, 0, nil, "i6", "a"),
			Valu("cmp_diff_2", OpLeq64, boolType, 0, nil, "two", "diff2"),
			If("cmp_diff_2", "b17", "b22")),

		// Path to heapify down loop
		Bloc("b17",
			Goto("b18")),

		// Path to heapify up check
		Bloc("b22",
			Goto("b16")),

		// Heapify up check: b - i >= 2
		Bloc("b16",
			Valu("mem16", OpPhi, memType, 0, nil, "mem14", "mem21"),
			Valu("diff3", OpSub64, intType, 0, nil, "b", "i6"),
			Valu("cmp_diff3_2", OpLeq64, boolType, 0, nil, "two", "diff3"),
			If("cmp_diff3_2", "b25", "b30")),

		// Start heapify up loop
		Bloc("b25",
			Valu("j_heapup_init", OpAdd64, intType, 0, nil, "i6", "one"),
			Goto("b26")),

		// Go to outer loop increment
		Bloc("b30",
			Goto("b4")),

		// Outer loop increment
		Bloc("b4",
			Valu("mem4", OpPhi, memType, 0, nil, "mem16", "mem29"),
			Valu("j_inc_outer", OpAdd64, intType, 0, nil, "j", "one"),
			Goto("b2")),

		// Heapify up loop header
		Bloc("b26",
			Valu("j_heapup", OpPhi, intType, 0, nil, "j_heapup_init", "j_heapup_inc"),
			Valu("mem26", OpPhi, memType, 0, nil, "mem16", "mem28"),
			Valu("cmp_b_j_heapup", OpLess64, boolType, 0, nil, "j_heapup", "b"),
			If("cmp_b_j_heapup", "b27", "b32")),

		// Call Less in heapify up
		Bloc("b27",
			Valu("less_ptr2", OpLoad, ptrType, 0, nil, "data_less", "mem26"),
			Valu("j_minus_1", OpSub64, intType, 0, nil, "j_heapup", "one"),
			Valu("call_less2", OpClosureCall, types.TypeResultMem, 0, nil, "less_ptr2", "data_less", "j_heapup", "j_minus_1", "mem26"),
			Valu("less_result2", OpSelectN, boolType, 0, nil, "call_less2"),
			Valu("mem27", OpSelectN, memType, 1, nil, "call_less2"),
			If("less_result2", "b28", "b31")),

		// Continue heapify up - call swap
		Bloc("b28",
			Valu("swap_ptr2", OpLoad, ptrType, 0, nil, "data_swap", "mem27"),
			Valu("call_swap2", OpClosureCall, memType, 0, nil, "swap_ptr2", "data_swap", "j_heapup", "j_minus_1", "mem27"),
			Valu("mem28", OpSelectN, memType, 0, nil, "call_swap2"),
			Valu("j_heapup_inc", OpAdd64, intType, 0, nil, "j_heapup", "one"),
			Goto("b26")),

		// Exit heapify up - not less
		Bloc("b31",
			Goto("b29")),

		// Exit heapify up - j >= b
		Bloc("b32",
			Goto("b29")),

		// Merge heapify up exits
		Bloc("b29",
			Valu("mem29", OpPhi, memType, 0, nil, "mem26", "mem27"),
			Goto("b4")),

		// Heapify down loop header
		Bloc("b18",
			Valu("j_heapdown", OpPhi, intType, 0, nil, "i_minus_1_b14", "j_heapdown_dec"),
			Valu("mem18", OpPhi, memType, 0, nil, "mem14", "mem20"),
			Valu("cmp_j_0", OpLess64, boolType, 0, nil, "zero_int", "j_heapdown"),
			If("cmp_j_0", "b19", "b7")),

		// Call Less in heapify down
		Bloc("b19",
			Valu("less_ptr3", OpLoad, ptrType, 0, nil, "data_less", "mem18"),
			Valu("j_heapdown_dec", OpSub64, intType, 0, nil, "j_heapdown", "one"),
			Valu("call_less3", OpClosureCall, types.TypeResultMem, 0, nil, "less_ptr3", "data_less", "j_heapdown", "j_heapdown_dec", "mem18"),
			Valu("less_result3", OpSelectN, boolType, 0, nil, "call_less3"),
			Valu("mem19", OpSelectN, memType, 1, nil, "call_less3"),
			If("less_result3", "b20", "b23")),

		// Continue heapify down - call swap
		Bloc("b20",
			Valu("swap_ptr3", OpLoad, ptrType, 0, nil, "data_swap", "mem19"),
			Valu("call_swap3", OpClosureCall, memType, 0, nil, "swap_ptr3", "data_swap", "j_heapdown", "j_heapdown_dec", "mem19"),
			Valu("mem20", OpSelectN, memType, 0, nil, "call_swap3"),
			Goto("b18")),

		// Exit heapify down - not less
		Bloc("b23",
			Goto("b21")),

		// Exit heapify down - j <= 0
		Bloc("b7",
			Goto("b21")),

		// Merge heapify down exits
		Bloc("b21",
			Valu("mem21", OpPhi, memType, 0, nil, "mem18", "mem19"),
			Goto("b16")),
	)

	return fun.f
}

func TestHeapSortCFGStructure(t *testing.T) {
	c := testConfig(t)
	f := buildHeapSortCFG(c)

	// Build block index for back-edge detection
	blockIndex := make(map[*Block]int)
	for i, b := range f.Blocks {
		blockIndex[b] = i
	}

	var (
		exitBlocks int
		ifBlocks   int
		gotoBlocks int
		phiCount   int
		backEdges  int
	)

	for _, b := range f.Blocks {
		switch len(b.Succs) {
		case 0:
			exitBlocks++
		case 1:
			gotoBlocks++
		case 2:
			ifBlocks++
		}

		for _, v := range b.Values {
			if v.Op == OpPhi {
				phiCount++
			}
		}

		for _, e := range b.Succs {
			if blockIndex[e.Block()] < blockIndex[b] {
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

	// Expected: 6 back-edges
	// - b4 -> b2 (outer j loop)
	// - b8 -> b6 (inner i/siftDown loop)
	// - b20 -> b18 (heapify down loop)
	// - b28 -> b26 (heapify up loop)
	// - b21 -> b16 (heapify down exit to heapify up check)
	// - b29 -> b4 (heapify up exit to outer loop)
	expectedBackEdges := 6
	if backEdges != expectedBackEdges {
		t.Errorf("expected %d back-edges, got %d", expectedBackEdges, backEdges)
	}

	// Expected: 3 exit blocks (b5, b13, b15)
	expectedExits := 3
	if exitBlocks != expectedExits {
		t.Errorf("expected %d exit blocks, got %d", expectedExits, exitBlocks)
	}
}

// buildFloatPrecCFG builds the CFG from math/big.(*Rat).FloatPrec
// This has 4 separate loops connected in sequence:
// - Bit-counting loop (b17 → b19 → b17)
// - Tab-building loop (b29 → b88 → b29)
// - Descending i-loop (b40 → b42 → b40)
// - p5 division loop (b65 → b66 → b65)
func buildFloatPrecCFG(c *Conf) *Func {
	ptrType := c.config.Types.BytePtr
	intType := c.config.Types.Int
	memType := types.TypeMem
	boolType := c.config.Types.Bool

	fun := c.Fun("b2",
		// Entry block
		Bloc("b2",
			Valu("x", OpArg, ptrType, 0, nil),
			Valu("mem", OpInitMem, memType, 0, nil),
			Valu("sp", OpSP, ptrType, 0, nil),
			Valu("sb", OpSB, ptrType, 0, nil),
			Valu("zero_int", OpConst64, intType, 0, nil),
			Valu("zero_bool", OpConstBool, boolType, 0, nil),
			Valu("one", OpConst64, intType, 1, nil),
			Valu("type_ptr", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("pool_result", OpStaticCall, ptrType, 0, nil, "mem"),
			Valu("mem2", OpStaticCall, memType, 0, nil, "mem"),
			Valu("cmp_type", OpEq64, boolType, 0, nil, "pool_result", "type_ptr"),
			If("cmp_type", "b3", "b4")),

		// Pool returned correct type
		Bloc("b3",
			Goto("b5")),

		// Pool returned nil/wrong type
		Bloc("b4",
			Valu("nil_stack", OpConstNil, ptrType, 0, nil),
			Goto("b5")),

		// Merge pool check
		Bloc("b5",
			Valu("stack", OpPhi, ptrType, 0, nil, "pool_result", "nil_stack"),
			Valu("has_stack", OpNeq64, boolType, 0, nil, "stack", "nil_stack"),
			If("has_stack", "b71", "b7")),

		// Has existing stack
		Bloc("b71",
			Goto("b8")),

		// Need to allocate stack
		Bloc("b7",
			Valu("new_stack", OpStaticCall, ptrType, 0, nil, "mem2"),
			Valu("mem7", OpStaticCall, memType, 0, nil, "mem2"),
			Goto("b8")),

		// Merge stack allocation
		Bloc("b8",
			Valu("mem8", OpPhi, memType, 0, nil, "mem2", "mem7"),
			Valu("stk", OpPhi, ptrType, 0, nil, "stack", "new_stack"),
			Valu("denom_len", OpLoad, intType, 0, nil, "x", "mem8"),
			Valu("denom_zero", OpEq64, boolType, 0, nil, "denom_len", "zero_int"),
			If("denom_zero", "b11", "b9")),

		// Denominator is zero - create unit denominator
		Bloc("b11",
			Valu("unit_ptr", OpStaticCall, ptrType, 0, nil, "mem8"),
			Valu("mem11", OpStaticCall, memType, 0, nil, "mem8"),
			Goto("b13")),

		// Denominator is non-zero
		Bloc("b9",
			Valu("denom_ptr", OpLoad, ptrType, 0, nil, "x", "mem8"),
			Goto("b13")),

		// Merge denominator
		Bloc("b13",
			Valu("mem13", OpPhi, memType, 0, nil, "mem11", "mem8"),
			Valu("d_ptr", OpPhi, ptrType, 0, nil, "unit_ptr", "denom_ptr"),
			Valu("d_len", OpLoad, intType, 0, nil, "d_ptr", "mem13"),
			Valu("d_len_zero", OpEq64, boolType, 0, nil, "d_len", "zero_int"),
			If("d_len_zero", "b15", "b14")),

		// d.len == 0: skip bit counting, go to main loop
		Bloc("b15",
			Valu("p2_init", OpConst64, intType, 0, nil),
			Goto("b26")),

		// d.len != 0: enter bit counting loop
		Bloc("b14",
			Valu("i_init", OpConst64, intType, 0, nil),
			Goto("b17")),

		// Bit counting loop header
		Bloc("b17",
			Valu("i_bit", OpPhi, intType, 0, nil, "i_init", "i_bit_inc"),
			Valu("cmp_i_len", OpLess64U, boolType, 0, nil, "i_bit", "d_len"),
			If("cmp_i_len", "b21", "b22")),

		// Check word for trailing zeros
		Bloc("b21",
			Valu("word", OpLoad, intType, 0, nil, "d_ptr", "mem13"),
			Valu("word_zero", OpEq64, boolType, 0, nil, "word", "zero_int"),
			If("word_zero", "b19", "b25")),

		// Word is zero, continue counting
		Bloc("b19",
			Valu("i_bit_inc", OpAdd64, intType, 0, nil, "i_bit", "one"),
			Goto("b17")),

		// Found non-zero word, compute trailing zeros
		Bloc("b25",
			Valu("tz_count", OpCtz64, intType, 0, nil, "word"),
			Valu("p2_computed", OpAdd64, intType, 0, nil, "tz_count", "i_bit"),
			Goto("b26")),

		// Bounds check failed (shouldn't happen)
		Bloc("b22",
			Exit("mem13")),

		// Main computation entry - start tab building
		Bloc("b26",
			Valu("p2", OpPhi, intType, 0, nil, "p2_init", "p2_computed"),
			Valu("q_ptr_init", OpStaticCall, ptrType, 0, nil, "mem13"),
			Valu("mem26", OpStaticCall, memType, 0, nil, "mem13"),
			Valu("f_ptr_init", OpStaticCall, ptrType, 0, nil, "mem26"),
			Valu("mem26b", OpStaticCall, memType, 0, nil, "mem26"),
			Goto("b29")),

		// Tab building loop header
		Bloc("b29",
			Valu("f_ptr", OpPhi, ptrType, 0, nil, "f_ptr_init", "f_ptr_sqr"),
			Valu("f_len", OpPhi, intType, 0, nil, "one", "f_len_sqr"),
			Valu("r_len", OpPhi, intType, 0, nil, "zero_int", "r_len_div"),
			Valu("r_ptr", OpPhi, ptrType, 0, nil, "q_ptr_init", "r_ptr_div"),
			Valu("mem29", OpPhi, memType, 0, nil, "mem26b", "mem88"),
			Valu("tab_ptr", OpPhi, ptrType, 0, nil, "q_ptr_init", "tab_ptr_new"),
			Valu("tab_len", OpPhi, intType, 0, nil, "zero_int", "tab_len_new"),
			Valu("tab_cap", OpPhi, intType, 0, nil, "zero_int", "tab_cap_new"),
			Valu("div_call", OpStaticCall, memType, 0, nil, "mem29"),
			Valu("r_len_div", OpSelectN, intType, 0, nil, "div_call"),
			Valu("r_ptr_div", OpSelectN, ptrType, 1, nil, "div_call"),
			Valu("mem29b", OpSelectN, memType, 2, nil, "div_call"),
			Valu("r_nonzero", OpNeq64, boolType, 0, nil, "r_len_div", "zero_int"),
			If("r_nonzero", "b31", "b32")),

		// r != 0: exit tab building, enter i-loop
		Bloc("b31",
			Valu("i_loop_init", OpSub64, intType, 0, nil, "tab_len", "one"),
			Goto("b40")),

		// r == 0: continue building tab
		Bloc("b32",
			Valu("tab_len_inc", OpAdd64, intType, 0, nil, "tab_len", "one"),
			Valu("need_grow", OpLess64U, boolType, 0, nil, "tab_cap", "tab_len_inc"),
			If("need_grow", "b34", "b51")),

		// No grow needed
		Bloc("b51",
			Goto("b30")),

		// Check if need to grow
		Bloc("b34",
			Valu("small_tab", OpLeq64, boolType, 0, nil, "tab_len_inc", "one"),
			If("small_tab", "b36", "b47")),

		// Small tab path
		Bloc("b36",
			Valu("already_alloc", OpLoad, boolType, 0, nil, "sp", "mem29b"),
			If("already_alloc", "b41", "b38")),

		// First allocation
		Bloc("b38",
			Valu("new_tab_small", OpStaticCall, ptrType, 0, nil, "mem29b"),
			Valu("mem38", OpStaticCall, memType, 0, nil, "mem29b"),
			Valu("cap_small", OpConst64, intType, 1, nil),
			Goto("b30")),

		// Already allocated
		Bloc("b41",
			Goto("b39")),

		// Large tab - need growslice
		Bloc("b47",
			Goto("b39")),

		// Grow slice
		Bloc("b39",
			Valu("grow_call", OpStaticCall, ptrType, 0, nil, "mem29b"),
			Valu("new_tab_large", OpSelectN, ptrType, 0, nil, "grow_call"),
			Valu("new_cap_large", OpSelectN, intType, 1, nil, "grow_call"),
			Valu("mem39", OpStaticCall, memType, 0, nil, "mem29b"),
			Goto("b30")),

		// Merge tab allocation paths
		Bloc("b30",
			Valu("tab_len_new", OpPhi, intType, 0, nil, "tab_len_inc", "tab_len_inc", "tab_len_inc"),
			Valu("tab_ptr_new", OpPhi, ptrType, 0, nil, "tab_ptr", "new_tab_small", "new_tab_large"),
			Valu("mem30", OpPhi, memType, 0, nil, "mem29b", "mem38", "mem39"),
			Valu("tab_cap_new", OpPhi, intType, 0, nil, "tab_cap", "cap_small", "new_cap_large"),
			Valu("wb_needed", OpLoad, boolType, 0, nil, "sb", "mem30"),
			If("wb_needed", "b89", "b85")),

		// No write barrier
		Bloc("b85",
			Goto("b88")),

		// Write barrier path
		Bloc("b89",
			Valu("wb_call", OpStaticCall, memType, 0, nil, "mem30"),
			Goto("b88")),

		// Square f and loop back
		Bloc("b88",
			Valu("mem88", OpPhi, memType, 0, nil, "mem30", "wb_call"),
			Valu("sqr_call", OpStaticCall, ptrType, 0, nil, "mem88"),
			Valu("f_ptr_sqr", OpSelectN, ptrType, 0, nil, "sqr_call"),
			Valu("f_len_sqr", OpSelectN, intType, 1, nil, "sqr_call"),
			Goto("b29")),

		// i-loop header (descending through tab)
		Bloc("b40",
			Valu("i_loop", OpPhi, intType, 0, nil, "i_loop_init", "i_loop_dec"),
			Valu("mem40", OpPhi, memType, 0, nil, "mem29b", "mem42"),
			Valu("p5", OpPhi, intType, 0, nil, "zero_int", "p5_new"),
			Valu("q_cap_40", OpPhi, intType, 0, nil, "tab_cap", "q_cap_42"),
			Valu("q_ptr_40", OpPhi, ptrType, 0, nil, "q_ptr_init", "q_ptr_42"),
			Valu("q_len_40", OpPhi, intType, 0, nil, "tab_len", "q_len_42"),
			Valu("i_nonneg", OpLeq64, boolType, 0, nil, "zero_int", "i_loop"),
			If("i_nonneg", "b44", "b43")),

		// Process tab[i]
		Bloc("b44",
			Valu("div_call2", OpStaticCall, memType, 0, nil, "mem40"),
			Valu("t_ptr", OpSelectN, ptrType, 0, nil, "div_call2"),
			Valu("t_len", OpSelectN, intType, 1, nil, "div_call2"),
			Valu("r2_len", OpSelectN, intType, 2, nil, "div_call2"),
			Valu("mem44", OpSelectN, memType, 3, nil, "div_call2"),
			Valu("r2_zero", OpEq64, boolType, 0, nil, "r2_len", "zero_int"),
			If("r2_zero", "b48", "b37")),

		// r2 != 0: skip to next iteration
		Bloc("b37",
			Goto("b42")),

		// r2 == 0: update p5 and possibly copy
		Bloc("b48",
			Valu("shift_amt", OpConst64, intType, 0, nil),
			Valu("p5_delta", OpLsh64x64, intType, 0, nil, "one", "i_loop"),
			Valu("p5_inc", OpAdd64, intType, 0, nil, "p5", "p5_delta"),
			Valu("need_copy", OpLess64, boolType, 0, nil, "q_cap_40", "t_len"),
			If("need_copy", "b50", "b54")),

		// No copy needed
		Bloc("b54",
			Goto("b56")),

		// Need to allocate new slice
		Bloc("b50",
			Valu("t_len_one", OpEq64, boolType, 0, nil, "t_len", "one"),
			If("t_len_one", "b58", "b57")),

		// Allocate size 1
		Bloc("b58",
			Valu("alloc1", OpStaticCall, ptrType, 0, nil, "mem44"),
			Valu("mem58", OpStaticCall, memType, 0, nil, "mem44"),
			Valu("cap1", OpConst64, intType, 1, nil),
			Goto("b56")),

		// Allocate larger
		Bloc("b57",
			Valu("new_cap", OpAdd64, intType, 0, nil, "t_len", "one"),
			Valu("alloc_large", OpStaticCall, ptrType, 0, nil, "mem44"),
			Valu("mem57", OpStaticCall, memType, 0, nil, "mem44"),
			Goto("b56")),

		// Merge allocation
		Bloc("b56",
			Valu("q_len_new", OpPhi, intType, 0, nil, "t_len", "one", "t_len"),
			Valu("q_ptr_new", OpPhi, ptrType, 0, nil, "q_ptr_40", "alloc1", "alloc_large"),
			Valu("mem56", OpPhi, memType, 0, nil, "mem44", "mem58", "mem57"),
			Valu("q_cap_new", OpPhi, intType, 0, nil, "q_cap_40", "cap1", "new_cap"),
			Valu("need_memmove", OpNeq64, boolType, 0, nil, "t_ptr", "q_ptr_new"),
			If("need_memmove", "b62", "b20")),

		// No memmove needed
		Bloc("b20",
			Goto("b63")),

		// Do memmove
		Bloc("b62",
			Valu("memmove_call", OpStaticCall, memType, 0, nil, "mem56"),
			Goto("b63")),

		// Merge memmove
		Bloc("b63",
			Valu("mem63", OpPhi, memType, 0, nil, "mem56", "memmove_call"),
			Goto("b42")),

		// i-loop increment (decrement)
		Bloc("b42",
			Valu("q_cap_42", OpPhi, intType, 0, nil, "q_cap_40", "q_cap_new"),
			Valu("q_len_42", OpPhi, intType, 0, nil, "q_len_40", "q_len_new"),
			Valu("p5_new", OpPhi, intType, 0, nil, "p5", "p5_inc"),
			Valu("q_ptr_42", OpPhi, ptrType, 0, nil, "q_ptr_40", "q_ptr_new"),
			Valu("mem42", OpPhi, memType, 0, nil, "mem44", "mem63"),
			Valu("i_loop_dec", OpSub64, intType, 0, nil, "i_loop", "one"),
			Goto("b40")),

		// Exit i-loop, enter p5 loop
		Bloc("b43",
			Goto("b65")),

		// p5 loop header (divide by 5)
		Bloc("b65",
			Valu("z_cap", OpPhi, intType, 0, nil, "q_cap_40", "z_cap_new"),
			Valu("z_ptr", OpPhi, ptrType, 0, nil, "q_ptr_40", "z_ptr_new"),
			Valu("mem65", OpPhi, memType, 0, nil, "mem40", "mem66"),
			Valu("z_len", OpPhi, intType, 0, nil, "q_len_40", "z_len_new"),
			Valu("p5_65", OpPhi, intType, 0, nil, "p5", "p5_65_inc"),
			Valu("div5_call", OpStaticCall, memType, 0, nil, "mem65"),
			Valu("t5_ptr", OpSelectN, ptrType, 0, nil, "div5_call"),
			Valu("t5_len", OpSelectN, intType, 1, nil, "div5_call"),
			Valu("r5_len", OpSelectN, intType, 2, nil, "div5_call"),
			Valu("mem65b", OpSelectN, memType, 3, nil, "div5_call"),
			Valu("r5_nonzero", OpNeq64, boolType, 0, nil, "r5_len", "zero_int"),
			If("r5_nonzero", "b67", "b68")),

		// r5 != 0: exit p5 loop, return
		Bloc("b67",
			Valu("cmp_result", OpStaticCall, intType, 0, nil, "mem65b"),
			Valu("mem67", OpStaticCall, memType, 0, nil, "mem65b"),
			// Use conditional select or just compute min with comparison
			Valu("p2_lt_p5", OpLess64U, boolType, 0, nil, "p2", "p5_65"),
			Valu("min_p", OpCondSelect, intType, 0, nil, "p2_lt_p5", "p2", "p5_65"),
			Valu("is_exact", OpEq64, boolType, 0, nil, "cmp_result", "zero_int"),
			Exit("mem67")),

		// r5 == 0: continue dividing
		Bloc("b68",
			Valu("need_copy2", OpLess64, boolType, 0, nil, "z_cap", "t5_len"),
			If("need_copy2", "b70", "b74")),

		// No copy needed
		Bloc("b74",
			Goto("b76")),

		// Need allocation
		Bloc("b70",
			Valu("t5_len_one", OpEq64, boolType, 0, nil, "t5_len", "one"),
			If("t5_len_one", "b78", "b77")),

		// Allocate size 1
		Bloc("b78",
			Valu("alloc1_p5", OpStaticCall, ptrType, 0, nil, "mem65b"),
			Valu("mem78", OpStaticCall, memType, 0, nil, "mem65b"),
			Valu("cap1_p5", OpConst64, intType, 1, nil),
			Goto("b76")),

		// Allocate larger
		Bloc("b77",
			Valu("new_cap_p5", OpAdd64, intType, 0, nil, "t5_len", "one"),
			Valu("alloc_large_p5", OpStaticCall, ptrType, 0, nil, "mem65b"),
			Valu("mem77", OpStaticCall, memType, 0, nil, "mem65b"),
			Goto("b76")),

		// Merge allocation
		Bloc("b76",
			Valu("z_len_phi", OpPhi, intType, 0, nil, "t5_len", "one", "t5_len"),
			Valu("z_ptr_phi", OpPhi, ptrType, 0, nil, "z_ptr", "alloc1_p5", "alloc_large_p5"),
			Valu("mem76", OpPhi, memType, 0, nil, "mem65b", "mem78", "mem77"),
			Valu("z_cap_phi", OpPhi, intType, 0, nil, "z_cap", "cap1_p5", "new_cap_p5"),
			Valu("need_memmove2", OpNeq64, boolType, 0, nil, "t5_ptr", "z_ptr_phi"),
			If("need_memmove2", "b82", "b87")),

		// No memmove needed
		Bloc("b87",
			Goto("b66")),

		// Do memmove
		Bloc("b82",
			Valu("memmove2_call", OpStaticCall, memType, 0, nil, "mem76"),
			Goto("b66")),

		// p5 loop increment
		Bloc("b66",
			Valu("mem66", OpPhi, memType, 0, nil, "mem76", "memmove2_call"),
			Valu("p5_65_inc", OpAdd64, intType, 0, nil, "p5_65", "one"),
			Valu("z_len_new", OpPhi, intType, 0, nil, "z_len_phi", "z_len_phi"),
			Valu("z_ptr_new", OpPhi, ptrType, 0, nil, "z_ptr_phi", "z_ptr_phi"),
			Valu("z_cap_new", OpPhi, intType, 0, nil, "z_cap_phi", "z_cap_phi"),
			Goto("b65")),
	)

	return fun.f
}

func TestFloatPrecCFGStructure(t *testing.T) {
	c := testConfig(t)
	f := buildFloatPrecCFG(c)

	blockIndex := make(map[*Block]int)
	for i, b := range f.Blocks {
		blockIndex[b] = i
	}

	var (
		exitBlocks int
		ifBlocks   int
		gotoBlocks int
		phiCount   int
		backEdges  int
	)

	for _, b := range f.Blocks {
		switch len(b.Succs) {
		case 0:
			exitBlocks++
		case 1:
			gotoBlocks++
		case 2:
			ifBlocks++
		}

		for _, v := range b.Values {
			if v.Op == OpPhi {
				phiCount++
			}
		}

		for _, e := range b.Succs {
			if blockIndex[e.Block()] < blockIndex[b] {
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

	// Expected: 4 back-edges
	// - b19 -> b17 (bit-counting loop)
	// - b88 -> b29 (tab-building loop)
	// - b42 -> b40 (i-loop)
	// - b66 -> b65 (p5-loop)
	expectedBackEdges := 4
	if backEdges != expectedBackEdges {
		t.Errorf("expected %d back-edges, got %d", expectedBackEdges, backEdges)
	}

	// Expected: 2 exit blocks (b22 panic, b67 return)
	expectedExits := 2
	if exitBlocks != expectedExits {
		t.Errorf("expected %d exit blocks, got %d", expectedExits, exitBlocks)
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

func BenchmarkComputeLive_HeapSort(b *testing.B) {
	c := testConfig(b)
	f := buildHeapSortCFG(c)
	benchmarkComputeLive(b, f)
}

func BenchmarkComputeLive_FloatPrec(b *testing.B) {
	c := testConfig(b)
	f := buildFloatPrecCFG(c)
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

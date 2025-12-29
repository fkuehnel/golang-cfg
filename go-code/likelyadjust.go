type loopnest struct {
	f              *Func
	b2l            []*loop    // block ID -> innermost containing loop
	po             []*Block   // cached postorder
	sdom           SparseTree // cached dominator tree (for compatibility)
	loops          []*loop    // all loops found
	hasIrreducible bool       // true if any irreducible loops detected
}

// loopnestfor computes loop nest information using Bourdoncle's algorithm.
//
// The algorithm:
//  1. Compute SCCs of the CFG (cached)
//  2. Each non-trivial SCC with single entry is a reducible loop; header = entry target
//  3. Remove header and recursively partition to find nested loops
//  4. Build loop tree based on containment
func loopnestfor(f *Func) *loopnest {
	po := f.postorder()
	b2l := make([]*loop, f.NumBlocks())
	loops := make([]*loop, 0)
	sawIrred := false

	if f.pass != nil && f.pass.debug > 2 {
		fmt.Printf("loop finding (Bourdoncle) in %s\n", f.Name)
	}

	sccs := f.sccs()
	if f.pass != nil && f.pass.debug > 3 {
		fmt.Printf("  found %d SCCs\n", len(sccs))
	}

	// Use cached top-level SCCs
	for i, scc := range sccs {
		if !scc.IsLoop() {
			continue
		}
		if !scc.IsReducible() {
			sawIrred = true
			continue
		}
		lscc := &sccs[i]
		if f.pass != nil && f.pass.debug > 3 {
			fmt.Printf("  processing loop SCC with %d blocks\n", len(lscc.Blocks))
		}
		// Recursively process this component
		processLoop(f, lscc, nil, b2l, &loops, &sawIrred)
	}

	// Compute nesting depths
	computeLoopDepths(loops)

	ln := &loopnest{
		f:              f,
		b2l:            b2l,
		po:             po,
		sdom:           nil, // We benchmark Bourdoncle, add later.
		loops:          loops,
		hasIrreducible: sawIrred,
	}

	if f.pass != nil && f.pass.debug > 1 && len(loops) > 0 {
		printLoopnest(f, b2l, loops)
	}
	// Curious about the loopiness? "-d=ssa/likelyadjust/stats"
	if f.pass != nil && f.pass.stats > 0 && len(loops) > 0 {
		logLoopStats(f, loops)
	}
	return ln
}

// processLoop recursively processes an SCC using Bourdoncle's decomposition.
func processLoop(f *Func, scc *SCC, outer *loop, b2l []*loop, loops *[]*loop, sawIrred *bool) {
	if len(scc.Blocks) == 0 {
		return
	}

	// Determine outermost header into SCC
	header := scc.Header()
	if header == nil {
		// Irreducible or whatnot -> not processing!
		*sawIrred = true
		if f.pass != nil && f.pass.debug > 3 {
			fmt.Printf("      header=%s (by dominance)\n", header)
		}
		return
	}

	// Create loop
	l := &loop{
		header:  header,
		outer:   outer,
		isInner: true,
		nBlocks: 1,
	}
	*loops = append(*loops, l)
	b2l[header.ID] = l

	// Mark outer as non-inner since it contains us
	if outer != nil {
		outer.isInner = false
	}

	// Collect non-header blocks
	remaining := make([]*Block, 0, len(scc.Blocks)-1)
	for _, b := range scc.Blocks {
		if b != header {
			remaining = append(remaining, b)
		}
	}

	if len(remaining) == 0 {
		if f.pass != nil && f.pass.debug > 3 {
			fmt.Printf("      no remaining blocks, done\n")
		}
		return
	}

	// Find nested SCCs with header removed
	if f.pass != nil && f.pass.debug > 3 {
		fmt.Printf("      remaining=%d, calling sccSubgraph\n", len(remaining))
	}
	subSccs := sccSubgraph(f, remaining, header)

	if f.pass != nil && f.pass.debug > 3 {
		fmt.Printf("      got %d sub-SCCs\n", len(subSccs))
		for j, sub := range subSccs {
			fmt.Printf("        sub[%d]: %d blocks, isLoop=%v\n",
				j, len(sub.Blocks), sub.IsLoop())
		}
	}

	for i := range subSccs {
		sub := &subSccs[i]
		if sub.IsLoop() {
			if !sub.IsReducible() {
				*sawIrred = true
			}
			// Nested loop
			processLoop(f, sub, l, b2l, loops, sawIrred)
		} else {
			// Trivial SCC: blocks belong to current loop
			for _, b := range sub.Blocks {
				if b2l[b.ID] == nil {
					b2l[b.ID] = l
					l.nBlocks++
				}
			}
		}
	}
}

// computeLoopDepths calculates nesting depth for all loops.
func computeLoopDepths(loops []*loop) {
	for _, l := range loops {
		if l.depth != 0 {
			// Already computed because it is an ancestor of
			// a previous loop.
			continue
		}
		// Find depth by walking up the loop tree.
		d := int16(0)
		for x := l; x != nil; x = x.outer {
			if x.depth != 0 {
				d += x.depth
				break
			}
			d++
		}
		// Set depth for every ancestor.
		for x := l; x != nil; x = x.outer {
			if x.depth != 0 {
				break
			}
			x.depth = d
			d--
		}
	}
	// Double-check depths.
	for _, l := range loops {
		want := int16(1)
		if l.outer != nil {
			want = l.outer.depth + 1
		}
		if l.depth != want {
			l.header.Fatalf("bad depth calculation for loop %s: got %d want %d", l.header, l.depth, want)
		}
	}
}

func printLoopnest(f *Func, b2l []*loop, loops []*loop) {
	fmt.Printf("Loops in %s:\n", f.Name)
	for _, l := range loops {
		fmt.Printf("%s, b=", l.LongString())
		for _, b := range f.Blocks {
			if b2l[b.ID] == l {
				fmt.Printf(" %s", b)
			}
		}
		fmt.Print("\n")
	}
	fmt.Printf("Nonloop blocks in %s:", f.Name)
	for _, b := range f.Blocks {
		if b2l[b.ID] == nil {
			fmt.Printf(" %s", b)
		}
	}
	fmt.Print("\n")
}

func logLoopStats(f *Func, loops []*loop) {

	// Note stats for non-innermost loops are slightly flawed because
	// they don't account for inner loop exits that span multiple levels.

	for _, l := range loops {
		inner := 0
		if l.isInner {
			inner++
		}

		f.LogStat("loopstats in "+f.Name+":",
			l.depth, "depth",
			inner, "is_inner", l.nBlocks, "n_blocks")
	}
}

// depth returns the loop nesting level of block b.
func (ln *loopnest) depth(b ID) int16 {
	if l := ln.b2l[b]; l != nil {
		return l.depth
	}
	return 0
}
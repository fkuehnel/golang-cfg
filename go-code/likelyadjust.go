type loop struct {
	header *Block // The header node of this (reducible) loop
	outer  *loop  // loop containing this loop

	// Next three fields used by regalloc and/or
	// aid in computation of inner-ness and list of blocks.
	nBlocks int32 // Number of blocks in this loop but not within inner loops
	depth   int16 // Nesting depth of the loop; 1 is outermost.
	isInner bool  // True if never discovered to contain a loop

	// True if all paths through the loop have a call.
	// Computed and used by regalloc; stored here for convenience.
	containsUnavoidableCall bool
}

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

	// Get top-level SCCs (cached via f.sccs())
	sccs := f.sccs()
	debug := f.pass != nil && f.pass.debug > 3

	if debug {
		fmt.Printf("  found %d SCCs\n", len(sccs))
	}

	// Create workspace once, reuse for all recursive decomposition
	work := newSCCWork(f.NumBlocks())

	for i := range sccs {
		scc := &sccs[i]
		if !scc.IsLoop() {
			continue
		}
		if !scc.IsReducible() {
			sawIrred = true
			continue
		}
		if debug {
			fmt.Printf("  processing loop SCC with %d blocks\n", len(scc.Blocks))
		}
		processLoop(f, work, scc, nil, b2l, &loops, &sawIrred, debug)
	}

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
func processLoop(f *Func, w *sccWork, scc *SCC, outer *loop, b2l []*loop, loops *[]*loop, sawIrred *bool, debug bool) {
	if len(scc.Blocks) == 0 {
		return
	}

	// Determine outermost header into SCC
	header := scc.Header()
	if header == nil {
		// Irreducible or whatnot -> not processing!
		*sawIrred = true
		if debug {
			fmt.Printf("      no header (irreducible)\n")
		}
		return
	}

	if debug {
		fmt.Printf("      header=%s\n", header)
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
		return
	}

	// Find nested SCCs with header removed
	if debug {
		fmt.Printf("      remaining=%d, decomposing\n", len(remaining))
	}
	subSccs := w.sccSubgraph(f, remaining, header)

	if debug {
		fmt.Printf("      got %d sub-SCCs\n", len(subSccs))
	}

	for i := range subSccs {
		sub := &subSccs[i]
		if sub.IsLoop() {
			if !sub.IsReducible() {
				*sawIrred = true
			}
			processLoop(f, w, sub, l, b2l, loops, sawIrred, debug)
		} else {
			// Trivial SCC: assign to current loop
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
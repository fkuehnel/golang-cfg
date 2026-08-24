type liveInfo struct {
	ID   ID       // ID of value
	dist int32    // # of instructions before next use
	pos  src.XPos // source position of next use
}

// computeLive computes a map from block ID to a list of value IDs live at the end
// of that block. Together with the value ID is a count of how many instructions
// to the next use of that value. The resulting map is stored in s.live.
func (s *regAllocState) computeLive() {
	f := s.f
	// single block functions do not have variables that are live across
	// branches
	if len(f.Blocks) == 1 {
		return
	}
	po := f.postorder()
	s.live = make([][]liveInfo, f.NumBlocks())
	s.desired = make([]desiredState, f.NumBlocks())
	s.loopnest = f.loopnest()

	rematIDs := make([]ID, 0, 64)

	live := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(live)
	t := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(t)

	s.loopnest.computeUnavoidableCalls()

	// Liveness analysis.
	// This is an adapted version of the algorithm described in chapter 2.4.2
	// of Fabrice Rastello's On Sparse Intermediate Representations.
	//   https://web.archive.org/web/20240417212122if_/https://inria.hal.science/hal-00761555/file/habilitation.pdf#section.50
	//
	// For our implementation, we fall back to a traditional iterative algorithm when we encounter
	// Irreducible CFGs. They are very uncommon in Go code because they need to be constructed with
	// gotos and our current loopnest definition does not compute all the information that
	// we'd need to compute the loop ancestors for that step of the algorithm.
	//
	// Additionally, instead of only considering non-loop successors in the initial DFS phase,
	// we compute the liveout as the union of all successors. This larger liveout set is a subset
	// of the final liveout for the block and adding this information in the DFS phase means that
	// we get slightly more accurate distance information.
	var loopLiveIn map[*loop][]liveInfo
	var numCalls []int32
	if len(s.loopnest.loops) > 0 && !s.loopnest.hasIrreducible {
		loopLiveIn = make(map[*loop][]liveInfo)
		numCalls = f.Cache.allocInt32Slice(f.NumBlocks())
		defer f.Cache.freeInt32Slice(numCalls)
	}

	for {
		changed := false

		for _, b := range po {
			// Start with known live values at the end of the block.
			live.clear()
			for _, e := range s.live[b.ID] {
				live.set(e.ID, e.dist, e.pos)
			}
			update := false
			// arguments to phi nodes are live at this blocks out
			for _, e := range b.Succs {
				succ := e.b
				delta := branchDistance(b, succ)
				for _, v := range succ.Values {
					if v.Op != OpPhi {
						break
					}
					arg := v.Args[e.i]
					if s.values[arg.ID].needReg && (!live.contains(arg.ID) || delta < live.get(arg.ID)) {
						live.set(arg.ID, delta, v.Pos)
						update = true
					}
				}
			}
			if update {
				s.live[b.ID] = updateLive(live, s.live[b.ID])
			}
			// Add len(b.Values) to adjust from end-of-block distance
			// to beginning-of-block distance.
			c := live.contents()
			for i := range c {
				c[i].val += int32(len(b.Values))
			}

			// Mark control values as live
			for _, c := range b.ControlValues() {
				if s.values[c.ID].needReg {
					live.set(c.ID, int32(len(b.Values)), b.Pos)
				}
			}

			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				live.remove(v.ID)
				if v.Op == OpPhi {
					continue
				}
				if opcodeTable[v.Op].call {
					if numCalls != nil {
						numCalls[b.ID]++
					}
					rematIDs = rematIDs[:0]
					c := live.contents()
					for i := range c {
						c[i].val += unlikelyDistance
						vid := c[i].key
						if s.values[vid].rematerializeable {
							rematIDs = append(rematIDs, vid)
						}
					}
					// We don't spill rematerializeable values, and assuming they
					// are live across a call would only force shuffle to add some
					// (dead) constant rematerialization. Remove them.
					for _, r := range rematIDs {
						live.remove(r)
					}
				}
				for _, a := range v.Args {
					if s.values[a.ID].needReg {
						live.set(a.ID, int32(i), v.Pos)
					}
				}
			}
			// This is a loop header, save our live-in so that
			// we can use it to fill in the loop bodies later
			if loopLiveIn != nil {
				loop := s.loopnest.b2l[b.ID]
				if loop != nil && loop.header.ID == b.ID {
					loopLiveIn[loop] = updateLive(live, nil)
				}
			}
			// For each predecessor of b, expand its list of live-at-end values.
			// invariant: live contains the values live at the start of b
			for _, e := range b.Preds {
				p := e.b
				delta := branchDistance(p, b)

				// Start t off with the previously known live values at the end of p.
				t.clear()
				for _, e := range s.live[p.ID] {
					t.set(e.ID, e.dist, e.pos)
				}
				update := false

				// Add new live values from scanning this block.
				for _, e := range live.contents() {
					d := e.val + delta
					if !t.contains(e.key) || d < t.get(e.key) {
						update = true
						t.set(e.key, d, e.pos)
					}
				}

				if !update {
					continue
				}
				s.live[p.ID] = updateLive(t, s.live[p.ID])
				changed = true
			}
		}

		// Doing a traditional iterative algorithm and have run
		// out of changes
		if !changed {
			break
		}

		// Doing a pre-pass and will fill in the liveness information
		// later
		if loopLiveIn != nil {
			break
		}
		// For loopless code, we have full liveness info after a single
		// iteration
		if len(s.loopnest.loops) == 0 {
			break
		}
	}
	if f.pass.debug > regDebug {
		s.debugPrintLive("after dfs walk", f, s.live, s.desired)
	}

	// irreducible CFGs and functions without loops are already
	// done, compute their desired registers and return
	if loopLiveIn == nil {
		s.computeDesired()
		return
	}

	// Walk the loopnest from outer to inner, adding
	// all live-in values from their parent. Instead of
	// a recursive algorithm, iterate in depth order.
	// TODO(dmo): can we permute the loopnest? can we avoid this copy?
	loops := slices.Clone(s.loopnest.loops)
	slices.SortFunc(loops, func(a, b *loop) int {
		return cmp.Compare(a.depth, b.depth)
	})

	loopset := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(loopset)
	for _, loop := range loops {
		if loop.outer == nil {
			continue
		}
		livein := loopLiveIn[loop]
		loopset.clear()
		for _, l := range livein {
			loopset.set(l.ID, l.dist, l.pos)
		}
		update := false
		for _, l := range loopLiveIn[loop.outer] {
			if !loopset.contains(l.ID) {
				loopset.set(l.ID, l.dist, l.pos)
				update = true
			}
		}
		if update {
			loopLiveIn[loop] = updateLive(loopset, livein)
		}
	}
	// unknownDistance is a sentinel value for when we know a variable
	// is live at any given block, but we do not yet know how far until it's next
	// use. The distance will be computed later.
	const unknownDistance = -1

	// add live-in values of the loop headers to their children.
	// This includes the loop headers themselves, since they can have values
	// that die in the middle of the block and aren't live-out
	for _, b := range po {
		loop := s.loopnest.b2l[b.ID]
		if loop == nil {
			continue
		}
		headerLive := loopLiveIn[loop]
		loopset.clear()
		for _, l := range s.live[b.ID] {
			loopset.set(l.ID, l.dist, l.pos)
		}
		update := false
		for _, l := range headerLive {
			if !loopset.contains(l.ID) {
				loopset.set(l.ID, unknownDistance, src.NoXPos)
				update = true
			}
		}
		if update {
			s.live[b.ID] = updateLive(loopset, s.live[b.ID])
		}
	}
	if f.pass.debug > regDebug {
		s.debugPrintLive("after live loop prop", f, s.live, s.desired)
	}
	// Filling in liveness from loops leaves some blocks with no distance information
	// Run over them and fill in the information from their successors.
	// To stabilize faster, we quit when no block has missing values and we only
	// look at blocks that still have missing values in subsequent iterations
	unfinishedBlocks := f.Cache.allocBlockSlice(len(po))
	defer f.Cache.freeBlockSlice(unfinishedBlocks)
	copy(unfinishedBlocks, po)

	for len(unfinishedBlocks) > 0 {
		n := 0
		for _, b := range unfinishedBlocks {
			live.clear()
			unfinishedValues := 0
			for _, l := range s.live[b.ID] {
				if l.dist == unknownDistance {
					unfinishedValues++
				}
				live.set(l.ID, l.dist, l.pos)
			}
			update := false
			for _, e := range b.Succs {
				succ := e.b
				for _, l := range s.live[succ.ID] {
					if !live.contains(l.ID) || l.dist == unknownDistance {
						continue
					}
					dist := int32(len(succ.Values)) + l.dist + branchDistance(b, succ)
					dist += numCalls[succ.ID] * unlikelyDistance
					val := live.get(l.ID)
					switch {
					case val == unknownDistance:
						unfinishedValues--
						fallthrough
					case dist < val:
						update = true
						live.set(l.ID, dist, l.pos)
					}
				}
			}
			if update {
				s.live[b.ID] = updateLive(live, s.live[b.ID])
			}
			if unfinishedValues > 0 {
				unfinishedBlocks[n] = b
				n++
			}
		}
		unfinishedBlocks = unfinishedBlocks[:n]
	}

	s.computeDesired()

	if f.pass.debug > regDebug {
		s.debugPrintLive("final", f, s.live, s.desired)
	}
}

// computeDesired computes the desired register information at the end of each block.
// It is essentially a liveness analysis on machine registers instead of SSA values
// The desired register information is stored in s.desired.
func (s *regAllocState) computeDesired() {

	// TODO: Can we speed this up using the liveness information we have already
	// from computeLive?
	// TODO: Since we don't propagate information through phi nodes, can we do
	// this as a single dominator tree walk instead of the iterative solution?
	var desired desiredState
	f := s.f
	po := f.postorder()
	for {
		changed := false
		for _, b := range po {
			desired.copy(&s.desired[b.ID])
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				prefs := desired.remove(v.ID)
				if v.Op == OpPhi {
					// TODO: if v is a phi, save desired register for phi inputs.
					// For now, we just drop it and don't propagate
					// desired registers back though phi nodes.
					continue
				}
				regspec := s.regspec(v)
				// Cancel desired registers if they get clobbered.
				desired.clobber(regspec.clobbers)
				// Update desired registers if there are any fixed register inputs.
				for _, j := range regspec.inputs {
					if countRegs(j.regs) != 1 {
						continue
					}
					desired.clobber(j.regs)
					desired.add(v.Args[j.idx].ID, pickReg(j.regs))
				}
				// Set desired register of input 0 if this is a 2-operand instruction.
				if opcodeTable[v.Op].resultInArg0 || v.Op == OpAMD64ADDQconst || v.Op == OpAMD64ADDLconst || v.Op == OpSelect0 {
					// ADDQconst is added here because we want to treat it as resultInArg0 for
					// the purposes of desired registers, even though it is not an absolute requirement.
					// This is because we'd rather implement it as ADDQ instead of LEAQ.
					// Same for ADDLconst
					// Select0 is added here to propagate the desired register to the tuple-generating instruction.
					if opcodeTable[v.Op].commutative {
						desired.addList(v.Args[1].ID, prefs)
					}
					desired.addList(v.Args[0].ID, prefs)
				}
			}
			for _, e := range b.Preds {
				p := e.b
				changed = s.desired[p.ID].merge(&desired) || changed
			}
		}
		if !changed || (!s.loopnest.hasIrreducible && len(s.loopnest.loops) == 0) {
			break
		}
	}
}

// updateLive updates a given liveInfo slice with the contents of t
func updateLive(t *sparseMapPos, live []liveInfo) []liveInfo {
	live = live[:0]
	if cap(live) < t.size() {
		live = make([]liveInfo, 0, t.size())
	}
	for _, e := range t.contents() {
		live = append(live, liveInfo{e.key, e.val, e.pos})
	}
	return live
}

// branchDistance calculates the distance between a block and a
// successor in pseudo-instructions. This is used to indicate
// likeliness
func branchDistance(b *Block, s *Block) int32 {
	if len(b.Succs) == 2 {
		if b.Succs[0].b == s && b.Likely == BranchLikely ||
			b.Succs[1].b == s && b.Likely == BranchUnlikely {
			return likelyDistance
		}
		if b.Succs[0].b == s && b.Likely == BranchUnlikely ||
			b.Succs[1].b == s && b.Likely == BranchLikely {
			return unlikelyDistance
		}
	}
	// Note: the branch distance must be at least 1 to distinguish the control
	// value use from the first user in a successor block.
	return normalDistance
}
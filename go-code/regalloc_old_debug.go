type liveInfo struct {
	ID   ID       // ID of value
	dist int32    // # of instructions before next use
	pos  src.XPos // source position of next use
}

// computeLive computes a map from block ID to a list of value IDs live at the end
// of that block. Together with the value ID is a count of how many instructions
// to the next use of that value. The resulting map is stored in s.live.
// computeLive also computes the desired register information at the end of each block.
// This desired register information is stored in s.desired.
// TODO: this could be quadratic if lots of variables are live across lots of
// basic blocks. Figure out a way to make this function (or, more precisely, the user
// of this function) require only linear size & time.
func (s *regAllocState) computeLive() {
	f := s.f
	s.live = make([][]liveInfo, f.NumBlocks())
	s.desired = make([]desiredState, f.NumBlocks())
	var phis []*Value

	live := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(live)
	t := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(t)

	// TODO: delete
	liveuseb := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(liveuseb)
	phisuseb := f.newSparseMapPos(f.NumValues())
	defer f.retSparseMapPos(phisuseb)
	liveout := make([][]liveInfo, f.NumBlocks())
	liveuse := make([][]liveInfo, f.NumBlocks())
	livedef := make([][]liveInfo, f.NumBlocks())
	phisref := make([][]liveInfo, f.NumBlocks())
	// end delete

	// Keep track of which value we want in each register.
	var desired desiredState

	// Calculate all strongly connected components of the CFG, and sort
	// the resulting DAG of SCC kernels in topological order.
	// We just traverse the topological sorted list once in reverse order, and
	// solve the data-flow equations (DFE) for live variable analysis in a given SCC
	// with the usual iterative approach.
	// TODO: iterating 3 times over blocks in the SCC in alternating post-order seems
	// to solve the DFEs in many cases. However, there are exceptions where it doesn't.
	// Is there a provable smarter way to solve DESs a given SCC?
	kernels := f.sccPartition()
	s.loopnest = f.loopnest()
	s.loopnest.calculateDepths()
	var passorder []*Block
	var maxIter = 0
	for j := len(kernels) - 1; j >= 0; j-- {
		scc := kernels[j] // scc entry block is first in list
		uporder, downorder := sccTraversalOrder(scc)
		iter := 0
		for iter = 0; ; iter++ {
			changed := false

			if iter&1 != 0 {
				passorder = downorder
			} else {
				passorder = uporder
			}
			if iter > maxIter {
				maxIter = iter
			}

			for _, b := range passorder {
				// TODO: delete
				liveuseb.clear()
				// Start with known live values at the end of the block.
				// Add len(b.Values) to adjust from end-of-block distance
				// to beginning-of-block distance.
				live.clear()
				for _, e := range s.live[b.ID] {
					live.set(e.ID, e.dist+int32(len(b.Values)), e.pos)
					// TODO: delete
					if iter == 1 {
						liveout[b.ID] = append(liveout[b.ID], e)
					}
					// end delete
				}

				// Mark control values as live
				for _, c := range b.ControlValues() {
					if s.values[c.ID].needReg {
						live.set(c.ID, int32(len(b.Values)), b.Pos)
						// TODO: delete
						if iter == 1 {
							liveuseb.set(c.ID, int32(len(b.Values)), b.Pos)
						}
						// end delete
					}
				}
				// Propagate backwards to the start of the block
				// Assumes Values have been scheduled.
				phis = phis[:0]
				for i := len(b.Values) - 1; i >= 0; i-- {
					v := b.Values[i]
					live.remove(v.ID)
					// TODO: delete
					if iter == 1 {
						liveuseb.remove(v.ID)
						livedef[b.ID] = append(livedef[b.ID], liveInfo{v.ID, int32(i), v.Pos})
					}
					// end delete
					if v.Op == OpPhi {
						// save phi ops for later
						phis = append(phis, v)
						continue
					}
					if opcodeTable[v.Op].call {
						c := live.contents()
						for i := range c {
							c[i].val += unlikelyDistance
						}
					}
					for _, a := range v.Args {
						if s.values[a.ID].needReg {
							live.set(a.ID, int32(i), v.Pos)
						}
					}
				}
				// TODO: delete
				if iter == 1 {
					liveuse[b.ID] = liveuse[b.ID][:0]
					for _, e := range liveuseb.contents() {
						liveuse[b.ID] = append(liveuse[b.ID], liveInfo{e.key, e.val, e.pos})
					}
				}
				// end delete

				// Propagate desired registers backwards.
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

				// For each predecessor of b, expand its list of live-at-end values.
				// invariant: live contains the values live at the start of b (excluding phi inputs)
				for i, e := range b.Preds {
					p := e.b
					// Compute additional distance for the edge.
					// Note: delta must be at least 1 to distinguish the control
					// value use from the first user in a successor block.
					delta := int32(normalDistance)
					if len(p.Succs) == 2 {
						if p.Succs[0].b == b && p.Likely == BranchLikely ||
							p.Succs[1].b == b && p.Likely == BranchUnlikely {
							delta = likelyDistance
						}
						if p.Succs[0].b == b && p.Likely == BranchUnlikely ||
							p.Succs[1].b == b && p.Likely == BranchLikely {
							delta = unlikelyDistance
						}
					}

					// Update any desired registers at the end of p.
					s.desired[p.ID].merge(&desired)

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
					// Also add the correct arg from the saved phi values.
					// All phis are at distance delta (we consider them
					// simultaneously happening at the start of the block).
					for _, v := range phis {
						id := v.Args[i].ID
						if s.values[id].needReg && (!t.contains(id) || delta < t.get(id)) {
							update = true
							t.set(id, delta, v.Pos)
						}
					}
					// TODO: delete
					if iter == 1 {
						phisref[p.ID] = phisref[p.ID][:0]
						for _, e := range phisuseb.contents() {
							phisref[p.ID] = append(phisref[p.ID], liveInfo{e.key, e.val, e.pos})
						}
					}
					// end delete

					if !update {
						continue
					}

					// The live set has changed, update it.
					l := s.live[p.ID][:0]
					if cap(l) < t.size() {
						l = make([]liveInfo, 0, t.size())
					}
					for _, e := range t.contents() {
						l = append(l, liveInfo{e.key, e.val, e.pos})
					}
					s.live[p.ID] = l
					changed = true
				}
			}

			if !changed {
				break
			}
		}
		// TODO: delete
		if iter > 4 && len(scc) < 15 {
			fmt.Printf("#iterations %d\n", iter)
			fmt.Printf("kernel: %v\n", scc)
			fmt.Printf("porder: %v\n", uporder)
			for _, b := range uporder {
				fmt.Printf("%s - %s:", b, b.LongString())
				for _, x := range s.live[b.ID] {
					fmt.Printf(" v%d(%d)", x.ID, x.dist)
				}
				fmt.Println()
			}
			fmt.Printf("out:\n")
			for _, b := range uporder {
				fmt.Printf("%s(%d): ", b, len(b.Values))
				for _, x := range liveout[b.ID] {
					fmt.Printf(" v%d(%d)", x.ID, x.dist)
				}
				fmt.Println()
			}
			fmt.Printf("refs:\n")
			for _, b := range uporder {
				fmt.Printf("%s(%d): ", b, len(b.Values))
				for _, x := range liveuse[b.ID] {
					fmt.Printf(" v%d(%d)", x.ID, x.dist)
				}
				fmt.Println()
			}
			fmt.Printf("phirefs:\n")
			for _, b := range uporder {
				fmt.Printf("%s(%d): ", b, len(b.Values))
				for _, x := range phisref[b.ID] {
					fmt.Printf(" v%d(%d)", x.ID, x.dist)
				}
				fmt.Println()
			}
			fmt.Printf("defs:\n")
			for _, b := range uporder {
				fmt.Printf("%s: ", b)
				for _, x := range livedef[b.ID] {
					fmt.Printf(" v%d(%d)", x.ID, x.dist)
				}
				fmt.Println()
			}
		}
		// end delete
	}
	if f.pass.debug > regDebug {
		fmt.Println("live values at end of each block")
		fmt.Printf("maximum iterations to solve DFEs: %d\n", maxIter)
		for _, b := range f.Blocks {
			fmt.Printf("  %s:", b)
			for _, x := range s.live[b.ID] {
				fmt.Printf(" v%d(%d)", x.ID, x.dist)
				for _, e := range s.desired[b.ID].entries {
					if e.ID != x.ID {
						continue
					}
					fmt.Printf("[")
					first := true
					for _, r := range e.regs {
						if r == noRegister {
							continue
						}
						if !first {
							fmt.Printf(",")
						}
						fmt.Print(&s.registers[r])
						first = false
					}
					fmt.Printf("]")
				}
			}
			if avoid := s.desired[b.ID].avoid; avoid != 0 {
				fmt.Printf(" avoid=%v", s.RegMaskString(avoid))
			}
			fmt.Println()
		}
	}
}

// A desiredState represents desired register assignments.
type desiredState struct {
	// Desired assignments will be small, so we just use a list
	// of valueID+registers entries.
	entries []desiredStateEntry
	// Registers that other values want to be in.  This value will
	// contain at least the union of the regs fields of entries, but
	// may contain additional entries for values that were once in
	// this data structure but are no longer.
	avoid regMask
}
type desiredStateEntry struct {
	// (pre-regalloc) value
	ID ID
	// Registers it would like to be in, in priority order.
	// Unused slots are filled with noRegister.
	// For opcodes that return tuples, we track desired registers only
	// for the first element of the tuple (see desiredSecondReg for
	// tracking the desired register for second part of a tuple).
	regs [4]register
}

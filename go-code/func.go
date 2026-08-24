// A Func represents a Go func declaration (or function literal) and its body.
// This package compiles each Func independently.
// Funcs are single-use; a new Func must be created for every compiled function.
type Func struct {
	Config *Config     // architecture information
	Cache  *Cache      // re-usable cache
	fe     Frontend    // frontend state associated with this Func, callbacks into compiler frontend
	pass   *pass       // current pass information (name, options, etc.)
	Name   string      // e.g. NewFunc or (*Func).NumBlocks (no package prefix)
	Type   *types.Type // type signature of the function.
	Blocks []*Block    // unordered set of all basic blocks (note: not indexable by ID)
	Entry  *Block      // the entry basic block

	bid idAlloc // block ID allocator
	vid idAlloc // value ID allocator

	HTMLWriter     *HTMLWriter    // html writer, for debugging
	PrintOrHtmlSSA bool           // true if GOSSAFUNC matches, true even if fe.Log() (spew phase results to stdout) is false.  There's an odd dependence on this in debug.go for method logf.
	ruleMatches    map[string]int // number of times countRule was called during compilation for any given string
	ABI0           *abi.ABIConfig // A copy, for no-sync access
	ABI1           *abi.ABIConfig // A copy, for no-sync access
	ABISelf        *abi.ABIConfig // ABI for function being compiled
	ABIDefault     *abi.ABIConfig // ABI for rtcall and other no-parsed-signature/pragma functions.

	maxCPUFeatures CPUfeatures // union of all the CPU features in all the blocks.

	scheduled   bool  // Values in Blocks are in final order
	laidout     bool  // Blocks are ordered
	NoSplit     bool  // true if function is marked as nosplit.  Used by schedule check pass.
	dumpFileSeq uint8 // the sequence numbers of dump file. (%s_%02d__%s.dump", funcname, dumpFileSeq, phaseName)
	IsPgoHot    bool
	DeferReturn *Block // avoid creating more than one deferreturn if there's multiple calls to deferproc-etc.

	// when register allocation is done, maps value ids to locations
	RegAlloc []Location

	// temporary registers allocated to rare instructions
	tempRegs map[ID]*Register

	// map from LocalSlot to set of Values that we want to store in that slot.
	NamedValues map[LocalSlot][]*Value
	// Names is a copy of NamedValues.Keys. We keep a separate list
	// of keys to make iteration order deterministic.
	Names []*LocalSlot
	// Canonicalize root/top-level local slots, and canonicalize their pieces.
	// Because LocalSlot pieces refer to their parents with a pointer, this ensures that equivalent slots really are equal.
	CanonicalLocalSlots  map[LocalSlot]*LocalSlot
	CanonicalLocalSplits map[LocalSlotSplitKey]*LocalSlot

	// RegArgs is a slice of register-memory pairs that must be spilled and unspilled in the uncommon path of function entry.
	RegArgs []Spill
	// OwnAux describes parameters and results for this function.
	OwnAux *AuxCall
	// CloSlot holds the compiler-synthesized name (".closureptr")
	// where we spill the closure pointer for range func bodies.
	CloSlot *ir.Name

	freeValues *Value // free Values linked by argstorage[0].  All other fields except ID are 0/nil.
	freeBlocks *Block // free Blocks linked by succstorage[0].b.  All other fields except ID are 0/nil.

	cachedPostorder  []*Block   // cached postorder traversal
	cachedIdom       []*Block   // cached immediate dominators
	cachedSdom       SparseTree // cached dominator tree
	cachedLoopnest   *loopnest  // cached loop nest information
	cachedSCCs       []SCC      // cached SCCs with entry information
	cachedLineStarts *xposmap   // cached map/set of xpos to integers

	auxmap    auxmap             // map from aux values to opaque ids used by CSE
	constants map[int64][]*Value // constants cache, keyed by constant value; users must check value's Op and Type
}

// sccs returns the cached SCCs for f, computing if necessary.
func (f *Func) sccs() []SCC {
	if f.cachedSCCs == nil {
		f.cachedSCCs = f.computeSCCs()
	}
	return f.cachedSCCs
}

// invalidateCFG tells f that its CFG has changed.
func (f *Func) invalidateCFG() {
	f.cachedPostorder = nil
	f.cachedIdom = nil
	f.cachedSdom = nil
	f.cachedLoopnest = nil
	f.cachedSCCs = nil
}
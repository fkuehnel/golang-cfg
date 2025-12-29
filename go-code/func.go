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
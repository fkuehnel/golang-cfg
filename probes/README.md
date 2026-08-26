# Probes

Small self-contained programs that settle a specific question. Each is runnable
with `go run .` and prints its own evidence. They exist because a claim in a code
review should be answerable by a command, not by an argument.

## `scc-order-coverage/`

**Question.** `computeLiveWithSccs` builds its second traversal order as
`exitward = dfsFrom(entryward[0])`. That is structurally identical to the
construction that fails in the Wolfram prototype, where the backward postorder
starts from an exit block and therefore covers one block out of eight. Does the
compiler inherit the defect?

**Answer: no.** The compiler's DFS is confined to the SCC (`if inSCC[succ.ID]`),
and inside a strongly connected component every block reaches every other, so
the second order is always total. The prototype ran over the whole CFG, where
reachability is not guaranteed.

```
compiler sccAlternatingOrdersDFS (confined to the SCC):
  SCCs checked (|scc|>=3): 5004
  orders failing to cover the SCC: 0

prototype construction (whole CFG, no SCC confinement):
  graphs checked: 20000
  second order failing to cover the first: 10318 (51.6%)
```

## `scc-convergence/`

**Question.** Coverage is only half of it. `computeLiveWithSccs` caps the
alternating sweeps at three and stops silently if values are still changing --
the "no guarantee" its own comment admits. Are three passes actually sufficient?

**Answer: not in general.** Simulating backward liveness over 200,000 random
strongly connected components:

```
passes needed to reach the fixed point:
   2 passes:    31434  (15.72%)
   3 passes:   134887  (67.44%)
   4 passes:    33520  (16.76%)
   5 passes:      157  ( 0.08%)
   6 passes:        2  ( 0.00%)

SCCs where the 3-pass cap yields a WRONG result: 159 (0.0795%)
  of those, UNDER-approximations (missing live values): 159
  of those, over-approximations (extra live values):      0
```

Note the two figures differ. Needing four passes usually means the values were
already correct after three and the fourth sweep merely confirms it. Only 0.08%
have values still changing after the third.

**The direction is the important part.** Every failure is an
under-approximation. The iteration starts from the empty set and grows
monotonically by union, so truncating it early always yields a subset of the
fixed point. A value that is live but not marked live can have its register
reused. If this ever fires in the compiler it is a miscompile, not a missed
optimisation.

### What this does and does not establish

It establishes that three alternating passes are **not sufficient in general**,
and that the failure mode is unsafe rather than merely suboptimal.

It does **not** establish that real Go CFGs reach the bad cases. The probe uses
random strongly connected digraphs with random def/use sets and models plain
bit-set liveness -- no distances, no rematerialisation, no desired-register
pass. Real Go control flow is mostly reducible with structured loops, and the
CL's own empirical claim is that no SCC in a 290,000-CFG sample needed more than
two passes. That is consistent with these cases being rare.

The open question is therefore not "is three enough" -- it is not -- but
"can a Go function produce an SCC that needs more than three", and if so,
whether the cap should be a convergence loop instead of a constant.

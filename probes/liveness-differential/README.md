# Liveness differential

Compares final liveness between two compiler builds by compiling a corpus with
`-gcflags=all=-d=ssa/regalloc/debug=3` and diffing the dumps.

```
go build -a -gcflags="all=-d=ssa/regalloc/debug=3" <pkgs> 2> live-A.txt   # arm A
go build -a -gcflags="all=-d=ssa/regalloc/debug=3" <pkgs> 2> live-B.txt   # arm B
python3 livediff.py live-A.txt live-B.txt
```

## Why not `analysis/compare.py`

`compare.py` pairs sections by label. The two arms here do not use the same
labels: the baseline emits `final:` and `after dfs walk:`, while the CL emits
`final (acyclic):`, `final (iterative):` and `final (SCC, N iterations):`. The
baseline also does not emit a final dump at all for loop-free functions, so the
two files cover different function sets -- 5463 `final` sections against 17801.

Run on the same pair of dumps:

| method | result |
|---|---|
| `livediff.py` | 0 differences across 16321 functions |
| `compare.py` | 9785 block-level differences |

The 9785 are artefacts of mismatched pairing. This matters beyond tooling
hygiene: `compare.py` is the script linked from the Gerrit review as the
evidence for the claim that the three-pass algorithm "produces smaller
distances", and that claim was drawn from exactly this kind of cross-label
comparison.

`compare.py` is not wrong in itself -- it is fine when both dumps use the same
labels. It is the wrong tool for comparing a patched compiler against its
baseline.

## What this one does

Keys sections by function name and takes each function's LAST section, which is
its final state whatever the label, then compares only functions present in
both. It reports value-set differences separately from distance differences,
because they answer different questions: a value-set difference is a
correctness bug, a distance difference is a register-allocation quality
question.

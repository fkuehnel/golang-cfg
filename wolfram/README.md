# Wolfram reference implementation

An independent implementation of Go SSA live-variable analysis, used as a
**differential oracle** against the Go compiler: parse a compiler dump,
recompute liveness from scratch, and compare. `FindDifference` returning `<||>`
means the two agree.

## Layout

```
LiveRange.wl        the package -- single source of truth for all definitions
corpus/<case>/      CFG dumps as data files (cfg, liveout, uses, phis, defs, entry)
tests/LiveRange.wlt regression tests
tests/run.wls       headless runner
notebooks/          the original exploration notebooks, preserved as-is
```

## Running the tests

Through a licensed kernel:

```wl
SetEnvironment["LIVERANGE_ROOT" -> "<abs path to this directory>"];
TestReport["tests/LiveRange.wlt"]
```

`wolframscript -file tests/run.wls` is the intended headless path.

Current status: **11 tests, 11 passing.**

## Why this package exists

`LiveRangeAnalysis.nb` and `ThreePassSuccess.nb` had independently diverged
copies of every function below. Worse, `ThreePassSuccess.nb` **cannot reproduce
its own saved results**: it opens with `Quit[]`, then calls `FindDifference`
(defined only in the other notebook) and `Source`/`Sink` (defined in neither).
Its `<||>` outputs came from a kernel whose state no longer exists.

The package is the canonical implementation. Notebooks should
`Get["LiveRange.wl"]` rather than redefine.

## Symbol mapping

| LiveRangeAnalysis.nb | ThreePassSuccess.nb | package |
|---|---|---|
| `postOrder[g,s][[2]]` | `postOrder[g,s][[2]]` | `PostOrder[g,s]` (returns the list directly) |
| `ParseGraph` | `ParseGraph` | `ParseCFG` |
| `ParseValues` | `ParseValues` | `ParseBlockValues` |
| `Valueize` | `Valuerize` | (internal) |
| `Blockize` | `Blockerize` | (internal) |
| `getValues` (safe lookup) | `fixMissing` | `LookupOrEmpty` |
| — | `getValues` (rule LHS) | `ValueNames` |
| — | `getDistances` | `ValueDistances` |
| `leavesFirst` | — | `LeavesFirst` |
| `passFunction` | `passFunction` | `PassFunction[revGraph, ctx]` |
| `IterativeSolution` | `IterativeSolution` | `IterativeSolution[start, g, entry, ctx]` |
| `ThreePassSolution` | `ThreePassSolution` | `ThreePassSolution[start, g, entry, ctx]` |
| `FindDifference` | (used, undefined) | `FindDifference` |
| `Source[node]` | (used, undefined) | `ctx["Uses"][node]` |
| `Sink[node]` | (used, undefined) | `ctx["Defs"][node]` |

Note the collision: `getValues` means *two different things* in the two
notebooks -- a missing-safe association lookup in one, rule-LHS extraction in
the other. They are separated here as `LookupOrEmpty` and `ValueNames`.

## Deliberate changes made during extraction

- **`Inactive[Blk][n]` / `Inactive[Val][n]`**, not `Inactive[b][n]` / `Inactive[v][n]`.
  Single-letter `b` and `v` collide with `Global`` in any interactive session;
  loading the package alongside a notebook that defines them produced
  `Symbol "b" appears in multiple contexts` shadow warnings, under which
  comparisons silently return wrong answers.
- **Value identities, not distances, in the solver.** `Union`, `Complement` and
  `SymmetricDifference` are set operations and are meaningless on
  `value -> distance` rules. Parsers keep distances; the solver strips them;
  `FindDifference` normalizes both sides before comparing.
- **Use/phi/def maps passed explicitly** as a context rather than read from
  globals. In the notebooks these were globals reassigned between examples, so
  one example's maps could leak into the next.

## Findings reproduced

Both from a clean kernel, asserted in the test suite:

| case | iterative | three-pass |
|---|---|---|
| `crypto-des` | matches compiler | matches compiler |
| `minimal-scc` | matches compiler | **wrong**: `b9` loses `v37`, `v41` |

**Three-pass is not correct in general.** Pass 2 walks the postorder from
`First[poFwd]`. On `minimal-scc` that is `b13`, an exit block with no
successors, so pass 2 covers **1 of 8** blocks and is effectively a no-op;
`b9` is never revisited. On `crypto-des` the same construction happens to cover
all 14 blocks, so the algorithm appears to work.

This confirms, on a real Go CFG, the mechanism `ThreePassSuccess.nb` identified
on the contrived irreducible 2-loop graph ("poBwd: Missing Nodes!"), and
explains the note in `LiveRangeAnalysis.nb`: *"Failure of the Three-pass
algorithm, something is not right it works here but not in the compiler."*

## Not done here (deliberately out of scope)

- The notebooks are **preserved unmodified**, not rewritten to use the package.
  They remain the historical record; new work should use the package.
- `LaTeXTikZ` / `ExportToTikZ` untouched. `ThreePassSuccess.nb` loads it via
  `PacletDirectoryLoad["<a directory outside the repository>"]`, an absolute path
  outside the repo, so the figure pipeline is not yet reproducible.
- `LiveRangeAnalysis.nb` is 1 MB, mostly embedded output graphics. Stripping
  outputs before committing future revisions would keep diffs sane.
- The corpus holds two hand-extracted cases. Extracting cases automatically from
  the archived full-size dumps is the next step
  if this is to run over hundreds of CFGs.

## A note on the notebooks

The original exploration notebooks are not included here. They cannot reproduce
their own saved output -- each opens with `Quit[]` and then calls functions it
does not define -- so shipping them would ship results no reader could rerun.
`LiveRange.wl` and the test suite are the reproducible form of that work; the
symbol mapping above records where each definition came from.

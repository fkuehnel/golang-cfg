# Hoare-Lomuto hybrid branchless quicksort

A Go port of the algorithm in Gerben Stavenga's *"Hoare's rebuttal and bubble
sort's comeback"* (`github.com/gerben-s/quicksort-blog-post`), following the
reference implementation in that repository's `hybrid_qsort.h`.

## What the algorithm is

The interesting claim is not "branchless Lomuto" -- that is Alexandrescu's.
It is that branchless Lomuto carries a **store-to-load dependency**: the loop
loads from the partition index and stores to it, the two addresses may alias,
so the CPU's memory disambiguation predictor refuses to speculate the load past
the store. A nominally one-cycle loop runs in about eight.

`distributeForward` removes the aliasing load by writing larger elements into a
scratch buffer instead of swapping: one load, one store, and the load can never
collide with a preceding store. That alone is not in place.

The **hybrid** restores in-place operation. Use a small fixed scratch. When it
fills, the region it just vacated becomes the scratch for a *backward*
distribution, filling toward the middle from the other end. Alternate until the
two ends meet. This is Hoare's outside-in structure driven by Lomuto's
distribute loop -- hence the blog's title -- and it is tunable: at
`ScratchSize == 1` it degenerates to plain Hoare.

The payoff is that branch mispredictions are amortised over `ScratchSize`
elements instead of paid per element.

## Layout

| function | role |
|---|---|
| `distributeForward` / `distributeBackward` | the branchless inner loops |
| `hybridPartition` | alternates the two, in place |
| `smallPartition` | single forward pass for inputs that fit in scratch |
| `choosePivotAndPartition` | median of three, plus an equal-element fallback for skewed splits |
| `bubbleSort2` | small-array sort, two elements bubbled per pass |
| `quickSort` | driver: recurse into the smaller side, heapsort on depth exhaustion |

## Correctness

`go test ./...` covers nine input distributions (random, sorted, reverse,
all-equal, few-unique, two-valued, sawtooth, organ-pipe, nearly sorted) at
sizes spanning the `ScratchSize` boundary.

Two properties are worth calling out:

- **Differential oracle.** `distributeForward` and `distributeBackward` use
  `unsafe` pointer arithmetic to get a single branchless store. The tests
  contain deliberately simple branchy reimplementations of both and assert that
  fast and safe agree exactly -- on the returned counts, the array contents,
  *and* the scratch contents.
- **The alternating loop is verified to execute.** Early in development the
  tests passed while exercising only the early-exit path, so this is checked
  explicitly rather than assumed.

The suite was mutation tested: eight deliberate defects were injected
(off-by-one on each partition boundary, inverted comparison, stalled scratch
counter, short final copy-back, wrong bubble stride, broken median). All were
caught. `medianOfThree` needed a dedicated test -- a bad pivot degrades
performance without breaking correctness, so no correctness test can see it.

## Notes on the Go port

- **Scratch is per call**, never package level, so `Sort` is safe under
  concurrent use. `TestSortConcurrent` runs it under `-race`.
- **No `uintptr` round trips.** Pointer arithmetic uses `unsafe.Add` on live
  `unsafe.Pointer` values, so the GC always sees a real pointer. `go vet` is
  clean; the `unsafeptr` check does not fire.
- Region writes are provably in bounds. Forward writes at `i+larger` with
  `larger <= 0`, never past the element just read. Backward writes at
  `j+smaller`, and every increment of `smaller` accompanies a decrement of `j`,
  so the index is non-increasing from its initial `n-1`.
- Inputs of at most `SmallSortThreshold` skip the scratch allocation entirely.

## Indicative timings

Laptop, `go1.26.4`, darwin/arm64, random `int`. **Not a measurement** -- this
machine is not quiet enough for small effects. The n=100000 gap is large enough
that noise is not the explanation; the n=1000 figures are not separable.

| n | this package | `slices.Sort` |
|---|---|---|
| 1000 | 8.4 ns/elem | 8.9 ns/elem |
| 100000 | 19.5 ns/elem | 51.8 ns/elem |

For scale, the C++ reference reports 16.4 ns/elem against `std::sort` at 51.6.

## Not done

- `SortFunc` with a caller-supplied comparator. A function call in the inner
  loop would defeat the branchless property; doing it properly needs generated
  code or a different interface.
- Tuning `ScratchSize` and `SmallSortThreshold`. Both are the reference's
  values, not measured on this port.
- Measurement on quiet hardware.

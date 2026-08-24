# Hybrid Hoare-Lomuto QuickSort - Go Implementation

Ported from [Google's C++ hybrid_qsort.h](https://github.com/google/...) (Apache 2.0 License)

## Files

- **gen_sort_variants_hybrid.go** - Generator with hybrid quicksort template
- **zsortordered.go** - Generated code for `cmp.Ordered` types
- **hybrid_sort_verify.go** - Correctness tests and benchmark

## Algorithm Overview

The hybrid quicksort combines two partitioning strategies:

| Approach | Pros | Cons |
|----------|------|------|
| **Hoare** | Symmetric, fewer swaps | Branch mispredictions per element |
| **Lomuto** | Can be branchless | 2 stores per iteration, memory dependencies |
| **Hybrid** | Best of both | Slightly more complex |

### Key Innovations

1. **Block-based partitioning**: Distributes elements in fixed-size blocks (64), reducing branch misses from N to N/64

2. **BubbleSort2**: Processes 2 elements per iteration for small arrays (~2 cycles inner loop)

3. **Skew detection**: When partition is <12.5% balanced, consolidates duplicate elements

4. **Tail recursion elimination**: Always recurses on smaller partition → O(log n) stack

## Benchmark Results (n=100,000)

```
Hybrid Sort: 8.5ms
std sort:    7.4ms  
Ratio:       1.15x
```

The Go version is slightly slower than `sort.Ints()` because:
- Go's pdqsort is highly optimized
- `sort.Interface` prevents element copying to scratch buffers (the key C++ optimization)
- Block partition overhead without cache benefits of element distribution

## Integration

Add the template code to `gen_sort_variants.go`:

```go
// In templateCode variable, append the hybrid sort functions
// Then call: hybridSort{{.FuncSuffix}}(data, 0, len(data) {{.ExtraArg}})
```

## Original C++ Performance (from README.md)

```
Random int's sorting:
std::sort           79 ns
exp_gerbens::QuickSort   30 ns   ← 2.6x faster!
pdqsort             42 ns
```

The C++ version achieves its speed via scratch buffers that enable true branchless distribution. The Go version preserves the algorithm structure but cannot fully replicate this due to interface constraints.

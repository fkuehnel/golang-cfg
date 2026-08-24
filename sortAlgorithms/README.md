<!-- framing note added 2026-08-24 -->
# Sorting algorithms

## Why this lives in the live-range repository

These are not a side project. They are the **workload** for the live-variable
analysis work in this repository, and they exercise it in two directions:

1. **As a corpus.** `go-code/README.md` drives SSA and register-allocator dumps
   through `qsort.go`; the `BenchmarkComputeLive_*` suite includes `HeapSort`
   and `FloatPrec`. Sorting code is small, loop-dense and register-hungry --
   exactly the shape that makes liveness and register allocation matter.

2. **As a symptom.** The central finding below is that Go does not emit
   conditional moves where C++ does, which forces the hand-written mask trick
   `dest = (addrLarge & mask) | (addrSmall &^ mask)`. CMOV emission is decided
   by the SSA backend and the register allocator -- the same subsystem the
   live-range work modifies. If that work changes which values are live across
   a branch, it can change whether a conditional move is generated at all.

So the sort benchmarks are a way to observe compiler behaviour end to end,
rather than only through the microbenchmarks in `go-code/`.

## Contents

| path | what |
|---|---|
| `hybrid/` | **current work.** Hoare-Lomuto hybrid branchless quicksort, its own Go module, tested and mutation tested. See `hybrid/README.md` |
| `branchless_final.go` | earlier standalone experiment; retains the `uintptr` pattern that `go vet` flags |
| `gen_sort_variants.go`, `gen_sort_variants_hybrid.go` | template-based generators, superseded by `hybrid/` |
| `hybrid_sort_verify.go` | correctness harness for the generated variants |
| `qsort.go`, `lomuto.go`, `qsort.cc` | reference partition schemes, used for SSA dumps |

`hybrid/` is a nested Go module, so `go test ./...` at this level does not
descend into it. Run it from `hybrid/`.

Status of every claim here is tracked in `../STATUS.md`.

---

# Branchless Conditional Moves in Go vs C++

In C++ the hot-loop using a branchless version of pointer assignment:
```cc
// From hybrid_qsort.h - DistributeForward
template <typename T, typename RandomIt, typename ScratchIt, typename Compare>
RandomIt DistributeForward(T pivot, RandomIt first, RandomIt last, 
                           ScratchIt scratch, ptrdiff_t scratch_size, Compare comp) {
  ptrdiff_t larger = 0;
  auto scratch_end = scratch + scratch_size - 1;
  
  while (first < last) {
    auto x = *first;                          // READ element into register
    bool is_larger = !comp(x, pivot);         // Compare (no branch yet!)
    auto dest = is_larger ? &scratch_end[larger] : &first[larger];  // Compute destination
    *dest = x;                                // WRITE to one of two places
    first++;
    larger -= is_larger;                      // Branchless counter update
    if (larger == -scratch_size) break;
  }
  return first + larger;
}
```

## Final Results: Unsafe Pointer Branchless QuickSort

```
=== Benchmark (n=100000, random data) ===

std sort (pdqsort):      7.47ms
BRANCHLESS unsafe ptr:   6.08ms  ← 18% FASTER!
standard Hoare:          6.98ms

Ratio branchless/std:    0.815x (18% faster!)
```

**The unsafe pointer branchless version beats Go's pdqsort on random data!**

---

## The Winning Pattern

```go
// Branchless pointer selection - generates optimal assembly
func distributeForward(data []int, lo, hi int, pivot int, scratch []int) int {
    // ...
    for i := 0; i < n; i++ {
        x := data[lo+i]
        
        // Branchless comparison to 0/1
        isLarger := 0
        if x >= pivot {
            isLarger = 1
        }
        mask := uintptr(-isLarger)  // 0 or 0xFFFFFFFF...
        
        // Compute both destinations
        addrSmall := baseData + uintptr(lo+smallerCount)*elemSize
        addrLarge := baseScratch + uintptr(scratchEnd-largerCount)*elemSize
        
        // Branchless select ONE destination
        dest := (addrLarge & mask) | (addrSmall &^ mask)
        
        // Single write through computed pointer!
        *(*int)(unsafe.Pointer(dest)) = x
        
        // Branchless counter updates
        smallerCount += 1 - isLarger
        largerCount += isLarger
    }
    return lo + smallerCount
}
```

---


## Generated Assembly (verified)

```bash
go tool compile -S cmov_simple.go 2>&1 | grep -E '(minBranch|minAssign|minBranchless|destIndex)' -A 15 | head -80
```

```asm
; Inner loop - NO conditional jumps!
SETLE   R8B              ; isLarger = (x >= pivot) ? 1 : 0
NEGQ    R8               ; mask = -isLarger
ANDQ    R8, R10          ; addrLarge & mask
NOTQ    R8               ; ~mask
ANDQ    R15, R8          ; addrSmall & ~mask
ORQ     R8, R10          ; dest = select(mask, large, small)
MOVQ    R13, (R10)       ; *dest = x  [SINGLE WRITE]
```

---

## Comparison: Three Approaches

| Approach | Random Data | Sorted | All Same |
|----------|-------------|--------|----------|
| std (pdqsort) | 7.47ms | 0.28ms | 0.08ms |
| **Branchless unsafe** | **6.08ms** | 7.49ms | 0.48ms |
| Standard Hoare | 6.98ms | 0.93ms | 2353ms! |

**Branchless wins on random** (the common case), but pdqsort's pattern detection wins on sorted/uniform data.

---

## Key Insights

### 1. Go CAN do true branchless with unsafe

```go
// This generates CMOV / branchless bit ops:
dest := (addrLarge & mask) | (addrSmall &^ mask)
*(*int)(unsafe.Pointer(dest)) = x
```

### 2. The pattern that triggers CMOV

```go
// ✅ CMOV pattern - assignment, not return
result := defaultValue
if condition {
    result = otherValue
}

// ❌ Branch pattern - early return
if condition {
    return x
}
return y
```

### 3. Bit manipulation for branchless selection

```go
mask := uintptr(-isLarger)  // 0 or -1 (all bits set)
selected := (a & mask) | (b &^ mask)  // branchless select
```

---

## Trade-offs

| | Branchless Unsafe | Standard |
|---|---|---|
| Random data | ✅ 18% faster | Baseline |
| Sorted data | ❌ 27x slower | Good |
| Duplicates | ✅ OK with skip | ❌ O(n²) |
| Safety | ❌ Requires unsafe | ✅ Safe |
| Portability | ⚠️ 64-bit int size | ✅ Portable |

---

## When to Use Branchless

**Use branchless unsafe pointer when:**
- Data is predominantly random
- Performance is critical
- You control the data type (int, not interface)

**Stick with pdqsort when:**
- Data might be partially sorted
- Safety is paramount
- You need pattern detection

## Problems with 8/16 bit data

```bash
go build -gcflags -S lomuto.go  # this doesn't cat into a file
go tool compile -S lomuto.go > lomuto.S
```

good compile with 64/32 bit word, double word:

```asm
	0x0030 00048 (lomuto.go:39)	MOVD	(R0)(R4<<3), R2
	...
	0x004c 00076 (lomuto.go:43)	MOVD	(R0)(R3<<3), R7
	0x0050 00080 (lomuto.go:45)	CMP	R2, R7
```

bad compile with 8/16 bit byte/half-word: (3 unnecessary operations)

```asm
	0x0030 00048 (lomuto.go:39)	MOVBU	(R0)(R4), R2
	...
	0x004c 00076 (lomuto.go:43)	MOVBU	(R0)(R3), R7
	0x0050 00080 (lomuto.go:45)	MOVD	R7, R8
	0x0054 00084 (lomuto.go:45)	MOVD	R2, R9
	0x0058 00088 (lomuto.go:45)	CMPW	R8, R2
	...
	0x0078 00120 (lomuto.go:45)	MOVD	R9, R2
```
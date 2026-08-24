// Package hybrid implements the Hoare-Lomuto hybrid branchless quicksort
// described in Gerben Stavenga's "Hoare's rebuttal and bubble sort's comeback"
// (github.com/gerben-s/quicksort-blog-post). It is a Go port of the reference
// implementation in that repository's hybrid_qsort.h.
//
// The central idea is not "branchless Lomuto" -- that is Alexandrescu's. It is
// that branchless Lomuto carries a store-to-load dependency: it loads from the
// partition index and stores to it, the addresses may alias, and so the CPU
// will not speculate the load past the store. That turns a nominally one-cycle
// loop into roughly eight.
//
// distributeForward removes the aliasing load by writing larger elements to a
// scratch buffer rather than swapping -- one load, one store, no aliasing. That
// alone is not in place. The hybrid restores in-place operation: use a small
// fixed scratch, and when it fills, the region it vacated becomes the scratch
// for a backward distribution. Alternate until the two ends meet. Hoare's
// outside-in structure driven by Lomuto's distribute loop.
package hybrid

import (
	"cmp"
	"math/bits"
	"unsafe"
)

const (
	// ScratchSize is the block size for branchless distribution. Larger values
	// amortise branch mispredictions over more elements; the buffer must stay
	// comfortably in L1.
	ScratchSize = 128

	// SmallSortThreshold is the size below which bubbleSort2 takes over.
	SmallSortThreshold = 16
)

// Sort sorts data in ascending order.
//
// The scratch buffer is allocated per call and never shared, so Sort is safe to
// call concurrently from multiple goroutines.
func Sort[T cmp.Ordered](data []T) {
	if len(data) < 2 {
		return
	}
	if len(data) <= SmallSortThreshold {
		bubbleSort2(data)
		return
	}
	scratch := make([]T, ScratchSize)
	quickSort(data, scratch, 2*bits.Len(uint(len(data))))
}

// ---------------------------------------------------------------------------
// distribution
// ---------------------------------------------------------------------------

// distributeForward walks a from the front. Elements less than pivot are
// compacted toward the front of a; elements greater than or equal to pivot are
// written to scratch, filling it backward from the end. It stops when scratch
// is full or a is exhausted.
//
// It returns the number of elements consumed and the number written to scratch.
// The partition point reached within a is consumed-nLarger.
//
// Writes into a are always at index i+larger with larger <= 0, hence never
// beyond the element just read; the region write is always in bounds.
func distributeForward[T cmp.Ordered](pivot T, a, scratch []T) (consumed, nLarger int) {
	n, ss := len(a), len(scratch)
	if n == 0 || ss == 0 {
		return 0, 0
	}
	var zero T
	esz := int(unsafe.Sizeof(zero))
	abase := unsafe.Pointer(&a[0])
	send := unsafe.Add(unsafe.Pointer(&scratch[0]), (ss-1)*esz)

	larger := 0 // non-positive: the running negative offset, as in the reference
	i := 0
	for i < n {
		x := a[i]
		// Assignment rather than early return: this is the shape the Go
		// compiler will lower to a conditional move.
		isLarger := 0
		if !(x < pivot) {
			isLarger = 1
		}
		dst := unsafe.Add(abase, i*esz)
		if isLarger != 0 {
			dst = send
		}
		*(*T)(unsafe.Add(dst, larger*esz)) = x
		i++
		larger -= isLarger
		if larger == -ss {
			break
		}
	}
	return i, -larger
}

// distributeBackward is distributeForward reversed: it walks a from the back,
// writing elements less than pivot into scratch filling it forward from the
// start, and compacting elements greater than or equal to pivot toward the end.
//
// Writes into a are at index j+smaller. Each increment of smaller accompanies a
// decrement of j, so j+smaller is non-increasing from its initial n-1 and the
// region write is always in bounds.
func distributeBackward[T cmp.Ordered](pivot T, a, scratch []T) (consumed, nSmaller int) {
	n, ss := len(a), len(scratch)
	if n == 0 || ss == 0 {
		return 0, 0
	}
	var zero T
	esz := int(unsafe.Sizeof(zero))
	abase := unsafe.Pointer(&a[0])
	sbase := unsafe.Pointer(&scratch[0])

	smaller := 0
	j := n
	for j > 0 {
		j--
		x := a[j]
		isSmaller := 0
		if x < pivot {
			isSmaller = 1
		}
		dst := unsafe.Add(abase, j*esz)
		if isSmaller != 0 {
			dst = sbase
		}
		*(*T)(unsafe.Add(dst, smaller*esz)) = x
		smaller += isSmaller
		if smaller == ss {
			break
		}
	}
	return n - j, smaller
}

// ---------------------------------------------------------------------------
// partition
// ---------------------------------------------------------------------------

// hybridPartition partitions a around pivot and returns p such that
// a[:p] are all < pivot and a[p:] are all >= pivot.
//
// scratch must have length ScratchSize and must not overlap a.
func hybridPartition[T cmp.Ordered](pivot T, a, scratch []T) int {
	ss := len(scratch)

	consumed, nLarger := distributeForward(pivot, a, scratch)
	pfirst := consumed - nLarger

	// Everything at or past pfirst is accounted for by scratch: copy it back
	// and we are done. scratch was filled backward, so the live entries are at
	// its tail.
	if size := len(a) - pfirst; size <= ss {
		copy(a[pfirst:], scratch[ss-size:ss])
		return pfirst
	}

	// [pfirst, pfirst+ss) now holds values already copied into scratch. It is
	// dead space, and becomes the scratch for the backward pass.
	first := pfirst + ss
	last := len(a)

	var res int
	for {
		c, nSmaller := distributeBackward(pivot, a[first:last], a[first-ss:first])
		last = last - c + nSmaller - ss
		if last <= first {
			res = last
			break
		}
		c2, nLarger2 := distributeForward(pivot, a[first:last], a[last:last+ss])
		first = first + c2 - nLarger2 + ss
		if last <= first {
			res = first - ss
			break
		}
	}
	copy(a[res:res+ss], scratch[:ss])
	return res
}

// smallPartition handles a <= ScratchSize by distributing forward once and
// copying the larger elements back. In place is unnecessary here because
// scratch is already big enough to hold every element that moves.
func smallPartition[T cmp.Ordered](pivot T, a, scratch []T) int {
	consumed, nLarger := distributeForward(pivot, a, scratch[:len(a)])
	p := consumed - nLarger
	copy(a[p:], scratch[len(a)-nLarger:len(a)])
	return p
}

// choosePivotAndPartition partitions a and returns (lo, hi): a[:lo] is strictly
// less than the pivot, a[hi:] is strictly greater, and a[lo:hi] is a block of
// elements equal to the pivot that needs no further sorting.
func choosePivotAndPartition[T cmp.Ordered](a, scratch []T) (int, int) {
	pivot := medianOfThree(a)

	var res int
	if len(a) > ScratchSize {
		res = hybridPartition(pivot, a, scratch)
	} else {
		res = smallPartition(pivot, a, scratch)
	}

	// A badly skewed split usually means a large run of elements equal to the
	// pivot. Isolate them so they are not re-partitioned forever.
	if res < len(a)>>3 {
		return res, res + partitionEqual(pivot, a[res:])
	}
	return res, res
}

// partitionEqual moves elements equal to pivot to the front of a, which is
// assumed to contain only elements >= pivot. It returns the count moved.
func partitionEqual[T cmp.Ordered](pivot T, a []T) int {
	k := 0
	for i := range a {
		if !(pivot < a[i]) { // a[i] <= pivot, and a[i] >= pivot, so equal
			a[i], a[k] = a[k], a[i]
			k++
		}
	}
	return k
}

// ---------------------------------------------------------------------------
// small sorts
// ---------------------------------------------------------------------------

// medianOfThree returns the median value of the first, middle and last element.
func medianOfThree[T cmp.Ordered](a []T) T {
	f, m, l := a[0], a[len(a)>>1], a[len(a)-1]
	if m < f {
		f, m = m, f
	}
	if l < f {
		f, l = l, f
	}
	if l < m {
		l, m = m, l
	}
	return m
}

// bubbleSort2 bubbles the two largest elements to the end on each pass, so it
// runs n(n+1)/4 inner iterations rather than n(n-1)/2. Ordering the conditional
// moves as below keeps the loop-carried dependency at two cycles, so halving
// the iteration count halves the running time even though the comparison count
// is unchanged.
//
// Insertion sort is the conventional choice here and is the wrong one: it
// mispredicts roughly once per insert. This loop has no unpredictable branches.
func bubbleSort2[T cmp.Ordered](a []T) {
	n := len(a)
	for i := n; i > 1; i -= 2 {
		x, y := a[0], a[1]
		if y < x {
			x, y = y, x
		}
		for j := 2; j < i; j++ {
			z := a[j]

			isSmaller := z < y
			w := y
			if isSmaller {
				w = z
			}
			if !isSmaller {
				y = z
			}

			isSmaller = z < x
			if isSmaller {
				a[j-2] = z
			} else {
				a[j-2] = x
				x = w
			}
		}
		a[i-2] = x
		a[i-1] = y
	}
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

func quickSort[T cmp.Ordered](a, scratch []T, depth int) {
	for len(a) > SmallSortThreshold {
		if depth == 0 {
			heapSort(a)
			return
		}
		depth--

		lo, hi := choosePivotAndPartition(a, scratch)

		// Recurse into the smaller side, iterate on the larger, so stack depth
		// stays O(log n).
		if lo < len(a)-hi {
			quickSort(a[:lo], scratch, depth)
			a = a[hi:]
		} else {
			quickSort(a[hi:], scratch, depth)
			a = a[:lo]
		}
	}
	bubbleSort2(a)
}

func heapSort[T cmp.Ordered](a []T) {
	n := len(a)
	for i := n/2 - 1; i >= 0; i-- {
		siftDown(a, i, n)
	}
	for i := n - 1; i > 0; i-- {
		a[0], a[i] = a[i], a[0]
		siftDown(a, 0, i)
	}
}

func siftDown[T cmp.Ordered](a []T, lo, hi int) {
	root := lo
	for {
		child := 2*root + 1
		if child >= hi {
			return
		}
		if child+1 < hi && a[child] < a[child+1] {
			child++
		}
		if !(a[root] < a[child]) {
			return
		}
		a[root], a[child] = a[child], a[root]
		root = child
	}
}

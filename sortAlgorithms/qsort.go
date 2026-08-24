// Inspect intermediate assembly
//  go build -gcflags -S qsort.go
//  go tool compile -S qsort.go > qsort.go.s
//
// Inspect SSA transformation
//  GOSSAFUNC=main.lomutoPartition go build qsort.go && open ssa.html
//
// Inspect final assembly
// go tool compile -S qsort.go > qsort0.s

package main

import (
    "fmt"
)

// Lomuto partition scheme for quicksort:
// https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme

func lomutoPartition(arr []uint8, lo uint, hi uint) uint {
	if hi-lo < 1 {
		return lo // nothing interesting to do
	}
	pivot := arr[hi]
	i := lo
	for j := lo; j < hi; j++ {
		if arr[j] < pivot {
			arr[i], arr[j] = arr[j], arr[i]
			i++
		}
	}
	arr[i], arr[hi] = arr[hi], arr[i]
	return i
}

func lomutoPartitionBranchFree(arr []uint8, lo uint, hi uint) uint {
	if hi-lo < 1 {
		return lo // nothing interesting to do
	}
	pivot := arr[hi]
	i := lo
	for j := lo; j < hi; j++ {
		// This is not possible with Go: less = -int(arr[j] < pivot)
		a, b := arr[i], arr[j]
		less := uint(0)
		if arr[j] < pivot { // This is compiled into a conditional move
			a, b, less = b, a, 1
			// i++ invalidates a condional move sequence
		}
		arr[i], arr[j] = a, b
		i += less
	}
	// Move the pivot to its final slot.
	arr[i], arr[hi] = arr[hi], arr[i]
	return i
}

func qsort(arr []uint8, lo uint, hi uint) {
	if lo < hi {
		p := lomutoPartitionBranchFree(arr, lo, hi)
		qsort(arr, lo, p-1)
		qsort(arr, p+1, hi)
	}
}

func main() {
    slice := []uint8{20, 8, 5, 1, 3, 11, 9, 7, 5, 10, 6, 2}
    qsort(slice, 0, uint(len(slice)-1))
    fmt.Println("\n--- Sorted ---\n\n", slice, "\n")
}
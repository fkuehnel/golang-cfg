package main

import ()

// Lomuto partition scheme for quicksort:
// https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme

func lomutoPartition(arr []uint8, first int, last int) int {
	if last-first < 1 {
		return first // nothing interesting to do
	}
	// Choose pivot, the smaler ones of arr[first] and arr[last]
	if arr[first] > arr[last] {
		arr[first], arr[last] = arr[last], arr[first]
	}
	pivot := arr[first]
	i := first
	// Prelude (not actually needed): position first (aka the write head)
	// right to the first larger element.
	for skip := true; skip; skip = (arr[i] < pivot) {
		i++
	}
	// Main loop
	for j := i + 1; j <= last; j++ {
		if arr[j] < pivot {
			arr[i], arr[j] = arr[j], arr[i]
			i++
		}
	}
	// Move the pivot to its final slot.
	arr[first], arr[i] = arr[i], arr[first]
	return i
}

func lomutoPartitionBranchFree(arr []uint8, first int, last int) int {
	if last-first < 1 {
		return first // nothing interesting to do
	}
	pivot := arr[last]
	i := first
	for j := first; j < last; j++ {
		// This is not possible with Go: less = -int(arr[j] < pivot)
		a, b := arr[i], arr[j]
		less := int(0)
		if arr[j] < pivot { // This is compiled into a conditional move
			a, b, less = b, a, 1
			// i++ invalidates a condional move sequence
		}
		arr[i], arr[j] = a, b
		i += less
	}
	// Move the pivot to its final slot.
	arr[i], arr[last] = arr[last], arr[i]
	return i
}

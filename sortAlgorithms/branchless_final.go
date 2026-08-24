package main

import (
	"fmt"
	"math/rand"
	"sort"
	"time"
	"unsafe"
)

// ============================================================================
// TRUE C++ STYLE BRANCHLESS QUICKSORT
// ============================================================================

var globalScratch = make([]int, 4096)

func ensureScratch(n int) []int {
	if n > len(globalScratch) {
		globalScratch = make([]int, n*2)
	}
	return globalScratch[:n]
}

// distributeForward: branchless distribution using unsafe pointer selection
func distributeForward(data []int, lo, hi int, pivot int, scratch []int) int {
	scratchSize := len(scratch)
	n := hi - lo
	if n == 0 || scratchSize == 0 {
		return lo
	}
	if n > scratchSize {
		n = scratchSize
	}

	smallerCount := 0
	largerCount := 0
	
	elemSize := uintptr(8)
	baseData := uintptr(unsafe.Pointer(&data[0]))
	baseScratch := uintptr(unsafe.Pointer(&scratch[0]))
	scratchEnd := scratchSize - 1
	
	for i := 0; i < n; i++ {
		x := data[lo+i]
		
		// Branchless comparison
		isLarger := 0
		if x >= pivot {
			isLarger = 1
		}
		mask := uintptr(-isLarger) // 0 or 0xFFFF...
		
		// Compute both destinations
		addrSmall := baseData + uintptr(lo+smallerCount)*elemSize
		addrLarge := baseScratch + uintptr(scratchEnd-largerCount)*elemSize
		
		// Select ONE destination branchlessly
		dest := (addrLarge & mask) | (addrSmall &^ mask)
		
		// Single write!
		*(*int)(unsafe.Pointer(dest)) = x
		
		// Branchless counter update
		smallerCount += 1 - isLarger
		largerCount += isLarger
	}
	
	return lo + smallerCount
}

// partitionBranchless: uses branchless distribution
func partitionBranchless(data []int, lo, hi int) int {
	n := hi - lo
	if n <= 1 {
		return lo
	}
	
	pivot := data[hi-1]
	scratch := ensureScratch(n)
	scratchEnd := len(scratch) - 1
	
	// Distribute elements (excluding pivot at hi-1)
	boundary := distributeForward(data, lo, hi-1, pivot, scratch)
	largerCount := (hi - 1 - lo) - (boundary - lo)
	
	// Place pivot at boundary
	data[boundary] = pivot
	
	// Copy larger elements after pivot
	for i := 0; i < largerCount; i++ {
		data[boundary+1+i] = scratch[scratchEnd-i]
	}
	
	return boundary
}

// partitionStandard: standard Hoare for comparison
func partitionStandard(data []int, lo, hi int) int {
	if hi-lo <= 1 {
		return lo
	}
	
	pivot := data[hi-1]
	left := lo
	right := hi - 2
	
	for left <= right {
		for left <= right && data[left] < pivot {
			left++
		}
		for left <= right && data[right] >= pivot {
			right--
		}
		if left < right {
			data[left], data[right] = data[right], data[left]
			left++
			right--
		}
	}
	
	data[left], data[hi-1] = data[hi-1], data[left]
	return left
}

// ============================================================================
// QUICKSORT IMPLEMENTATIONS
// ============================================================================

func medianOfThreeToEnd(data []int, lo, hi int) {
	mid := lo + (hi-lo)/2
	last := hi - 1
	if data[mid] < data[lo] {
		data[lo], data[mid] = data[mid], data[lo]
	}
	if data[last] < data[lo] {
		data[lo], data[last] = data[last], data[lo]
	}
	if data[last] < data[mid] {
		data[mid], data[last] = data[last], data[mid]
	}
	data[mid], data[last] = data[last], data[mid]
}

func insertionSort(data []int, lo, hi int) {
	for i := lo + 1; i < hi; i++ {
		for j := i; j > lo && data[j] < data[j-1]; j-- {
			data[j], data[j-1] = data[j-1], data[j]
		}
	}
}

func quicksortBranchless(data []int) {
	var impl func(lo, hi int)
	impl = func(lo, hi int) {
		for hi-lo > 16 {
			medianOfThreeToEnd(data, lo, hi)
			p := partitionBranchless(data, lo, hi)
			
			// Skip duplicates of pivot
			pEnd := p + 1
			for pEnd < hi && data[pEnd] == data[p] {
				pEnd++
			}
			
			// Recurse on smaller partition (for O(log n) stack)
			if p-lo < hi-pEnd {
				impl(lo, p)
				lo = pEnd
			} else {
				impl(pEnd, hi)
				hi = p
			}
		}
		insertionSort(data, lo, hi)
	}
	impl(0, len(data))
}

func quicksortStandard(data []int) {
	var impl func(lo, hi int)
	impl = func(lo, hi int) {
		for hi-lo > 16 {
			medianOfThreeToEnd(data, lo, hi)
			p := partitionStandard(data, lo, hi)
			
			if p-lo < hi-p-1 {
				impl(lo, p)
				lo = p + 1
			} else {
				impl(p+1, hi)
				hi = p
			}
		}
		insertionSort(data, lo, hi)
	}
	impl(0, len(data))
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	rand.Seed(time.Now().UnixNano())
	
	fmt.Println("=== Branchless Unsafe Pointer Quicksort ===\n")
	
	// Test correctness
	allPass := true
	for _, n := range []int{10, 100, 1000, 10000, 50000} {
		data := make([]int, n)
		for i := range data {
			data[i] = rand.Intn(n * 10)
		}
		
		expected := make([]int, n)
		copy(expected, data)
		sort.Ints(expected)
		
		quicksortBranchless(data)
		
		ok := true
		for i := range data {
			if data[i] != expected[i] {
				ok = false
				break
			}
		}
		
		if ok {
			fmt.Printf("✓ n=%d: PASS\n", n)
		} else {
			fmt.Printf("✗ n=%d: FAIL\n", n)
			allPass = false
		}
	}
	
	if !allPass {
		fmt.Println("\nTests failed!")
		return
	}
	
	// Benchmark
	fmt.Println("\n=== Benchmark (n=100000, 20 iterations) ===")
	
	n := 100000
	iterations := 20
	
	var stdTotal, branchlessTotal, standardTotal time.Duration
	
	for iter := 0; iter < iterations; iter++ {
		seed := rand.Int63()
		
		rand.Seed(seed)
		data1 := make([]int, n)
		for i := range data1 { data1[i] = rand.Intn(n * 10) }
		
		rand.Seed(seed)
		data2 := make([]int, n)
		for i := range data2 { data2[i] = rand.Intn(n * 10) }
		
		rand.Seed(seed)
		data3 := make([]int, n)
		for i := range data3 { data3[i] = rand.Intn(n * 10) }
		
		start := time.Now()
		sort.Ints(data1)
		stdTotal += time.Since(start)
		
		start = time.Now()
		quicksortBranchless(data2)
		branchlessTotal += time.Since(start)
		
		start = time.Now()
		quicksortStandard(data3)
		standardTotal += time.Since(start)
	}
	
	fmt.Printf("\nstd sort (pdqsort):      %v avg\n", stdTotal/time.Duration(iterations))
	fmt.Printf("BRANCHLESS unsafe ptr:   %v avg\n", branchlessTotal/time.Duration(iterations))
	fmt.Printf("standard Hoare:          %v avg\n", standardTotal/time.Duration(iterations))
	fmt.Printf("\nRatio branchless/std:    %.3fx\n", float64(branchlessTotal)/float64(stdTotal))
	fmt.Printf("Ratio standard/std:      %.3fx\n", float64(standardTotal)/float64(stdTotal))
	
	// Pattern tests
	fmt.Println("\n=== Data Patterns (n=100000) ===")
	
	patterns := []struct {
		name string
		gen  func(n int) []int
	}{
		{"Random", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = rand.Intn(n * 10) }
			return d
		}},
		{"Sorted", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = i }
			return d
		}},
		{"Reverse", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = n - i }
			return d
		}},
		{"All Same", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = 42 }
			return d
		}},
		{"Few Unique", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = rand.Intn(10) }
			return d
		}},
	}
	
	for _, p := range patterns {
		data := p.gen(n)
		data2 := make([]int, n)
		data3 := make([]int, n)
		copy(data2, data)
		copy(data3, data)
		
		start := time.Now()
		sort.Ints(data)
		stdTime := time.Since(start)
		
		start = time.Now()
		quicksortBranchless(data2)
		blTime := time.Since(start)
		
		start = time.Now()
		quicksortStandard(data3)
		stTime := time.Since(start)
		
		fmt.Printf("\n%s:\n", p.name)
		fmt.Printf("  std:        %v\n", stdTime)
		fmt.Printf("  branchless: %v (%.2fx)\n", blTime, float64(blTime)/float64(stdTime))
		fmt.Printf("  standard:   %v (%.2fx)\n", stTime, float64(stTime)/float64(stdTime))
	}
}

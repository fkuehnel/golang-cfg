// Test file for hybrid quicksort implementation
package main

import (
	"cmp"
	"fmt"
	"math/rand"
	"sort"
	"time"
)

// Copy the generated functions here for testing
const (
	hybridSmallSortThresholdOrdered = 16
	hybridBlockSizeOrdered          = 64
)

func insertionSortOrdered[E cmp.Ordered](data []E, a, b int) {
	for i := a + 1; i < b; i++ {
		for j := i; j > a && cmp.Less(data[j], data[j-1]); j-- {
			data[j], data[j-1] = data[j-1], data[j]
		}
	}
}

func hybridBubbleSort2Ordered[E cmp.Ordered](data []E, a, b int) {
	n := b - a
	if n <= 1 {
		return
	}
	for size := n; size > 1; size -= 2 {
		if size >= 2 && cmp.Less(data[a+1], data[a]) {
			data[a], data[a+1] = data[a+1], data[a]
		}
		for j := 2; j < size; j++ {
			curr := a + j
			prev := a + j - 1
			prev2 := a + j - 2
			if cmp.Less(data[curr], data[prev]) {
				data[curr], data[prev] = data[prev], data[curr]
				if j >= 2 && cmp.Less(data[prev], data[prev2]) {
					data[prev], data[prev2] = data[prev2], data[prev]
				}
			}
		}
	}
	for i := a + 1; i < b; i++ {
		for j := i; j > a && cmp.Less(data[j], data[j-1]); j-- {
			data[j], data[j-1] = data[j-1], data[j]
		}
	}
}

func hybridSmallSortOrdered[E cmp.Ordered](data []E, a, b int) {
	n := b - a
	if n <= 1 {
		return
	}
	if n <= 8 {
		insertionSortOrdered(data, a, b)
		return
	}
	hybridBubbleSort2Ordered(data, a, b)
}

func hybridMedianOfThreeToEndOrdered[E cmp.Ordered](data []E, a, b int) {
	mid := a + (b-a)/2
	last := b - 1
	if cmp.Less(data[mid], data[a]) {
		data[a], data[mid] = data[mid], data[a]
	}
	if cmp.Less(data[last], data[a]) {
		data[a], data[last] = data[last], data[a]
	}
	if cmp.Less(data[last], data[mid]) {
		data[mid], data[last] = data[last], data[mid]
	}
	data[mid], data[last] = data[last], data[mid]
}

func hybridHoarePartitionOrdered[E cmp.Ordered](data []E, a, b int) int {
	pivot := b - 1
	i := a
	j := pivot - 1
	for i <= j {
		for i <= j && cmp.Less(data[i], data[pivot]) {
			i++
		}
		for i <= j && !cmp.Less(data[j], data[pivot]) {
			j--
		}
		if i < j {
			data[i], data[j] = data[j], data[i]
			i++
			j--
		}
	}
	data[i], data[pivot] = data[pivot], data[i]
	return i
}

func hybridBlockPartitionOrdered[E cmp.Ordered](data []E, a, b int) int {
	if b-a <= 1 {
		return a
	}
	pivot := b - 1
	left := a
	right := pivot - 1
	for left <= right {
		blockEnd := left + hybridBlockSizeOrdered
		if blockEnd > right+1 {
			blockEnd = right + 1
		}
		writePos := left
		for readPos := left; readPos < blockEnd; readPos++ {
			if cmp.Less(data[readPos], data[pivot]) {
				if readPos != writePos {
					data[readPos], data[writePos] = data[writePos], data[readPos]
				}
				writePos++
			}
		}
		blockStart := right - hybridBlockSizeOrdered + 1
		if blockStart < writePos {
			blockStart = writePos
		}
		writeEnd := right
		for readPos := right; readPos >= blockStart; readPos-- {
			if !cmp.Less(data[readPos], data[pivot]) {
				if readPos != writeEnd {
					data[readPos], data[writeEnd] = data[writeEnd], data[readPos]
				}
				writeEnd--
			}
		}
		left = writePos
		right = writeEnd
		for left < right {
			for left < right && cmp.Less(data[left], data[pivot]) {
				left++
			}
			for left < right && !cmp.Less(data[right], data[pivot]) {
				right--
			}
			if left < right {
				data[left], data[right] = data[right], data[left]
				left++
				right--
			}
		}
	}
	data[left], data[pivot] = data[pivot], data[left]
	return left
}

func hybridChoosePivotAndPartitionOrdered[E cmp.Ordered](data []E, a, b int) (int, int) {
	hybridMedianOfThreeToEndOrdered(data, a, b)
	var p int
	if b-a > hybridBlockSizeOrdered*2 {
		p = hybridBlockPartitionOrdered(data, a, b)
	} else {
		p = hybridHoarePartitionOrdered(data, a, b)
	}
	n := b - a
	m := p - a
	if m < n/8 {
		eq := p
		for i := p + 1; i < b; i++ {
			if !cmp.Less(data[p], data[i]) && !cmp.Less(data[i], data[p]) {
				eq++
				if i != eq {
					data[i], data[eq] = data[eq], data[i]
				}
			}
		}
		if eq > p {
			return p, eq + 1
		}
	}
	return p, p
}

func hybridQuickSortImplOrdered[E cmp.Ordered](data []E, a, b int) {
	for b-a > hybridSmallSortThresholdOrdered {
		p, pEnd := hybridChoosePivotAndPartitionOrdered(data, a, b)
		nleft := p - a
		nright := b - pEnd
		if nleft <= nright {
			hybridQuickSortImplOrdered(data, a, p)
			a = pEnd
		} else {
			hybridQuickSortImplOrdered(data, pEnd, b)
			b = p
		}
	}
	hybridSmallSortOrdered(data, a, b)
}

func hybridSortOrdered[E cmp.Ordered](data []E, a, b int) {
	if b-a <= 1 {
		return
	}
	hybridQuickSortImplOrdered(data, a, b)
}

func isSorted[E cmp.Ordered](data []E) bool {
	for i := 1; i < len(data); i++ {
		if cmp.Less(data[i], data[i-1]) {
			return false
		}
	}
	return true
}

func main() {
	rand.Seed(time.Now().UnixNano())
	
	fmt.Println("=== Hybrid QuickSort Correctness Tests ===\n")
	
	tests := []struct {
		name string
		gen  func(n int) []int
		sizes []int
	}{
		{"Random", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = rand.Intn(n * 10) }
			return d
		}, []int{0, 1, 2, 3, 5, 10, 16, 17, 50, 100, 1000, 10000}},
		
		{"Already Sorted", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = i }
			return d
		}, []int{10, 100, 1000}},
		
		{"Reverse Sorted", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = n - i }
			return d
		}, []int{10, 100, 1000}},
		
		{"All Same", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = 42 }
			return d
		}, []int{10, 100, 1000}},
		
		{"Two Values", func(n int) []int {
			d := make([]int, n)
			for i := range d { d[i] = rand.Intn(2) }
			return d
		}, []int{10, 100, 1000}},
		
		{"Organ Pipe", func(n int) []int {
			d := make([]int, n)
			for i := 0; i < n/2; i++ { d[i] = i }
			for i := n/2; i < n; i++ { d[i] = n - i }
			return d
		}, []int{10, 100, 1000}},
	}
	
	passed := 0
	failed := 0
	
	for _, tc := range tests {
		for _, size := range tc.sizes {
			data := tc.gen(size)
			
			// Get expected result
			expected := make([]int, len(data))
			copy(expected, data)
			sort.Ints(expected)
			
			// Run hybrid sort
			hybridSortOrdered(data, 0, len(data))
			
			// Verify
			ok := true
			if !isSorted(data) {
				ok = false
			} else {
				for i := range data {
					if data[i] != expected[i] {
						ok = false
						break
					}
				}
			}
			
			if ok {
				fmt.Printf("✓ %s n=%d: PASS\n", tc.name, size)
				passed++
			} else {
				fmt.Printf("✗ %s n=%d: FAIL\n", tc.name, size)
				failed++
			}
		}
	}
	
	fmt.Printf("\n=== Results: %d passed, %d failed ===\n", passed, failed)
	
	// Quick benchmark comparison
	fmt.Println("\n=== Quick Benchmark (n=100000) ===")
	
	n := 100000
	data1 := make([]int, n)
	data2 := make([]int, n)
	for i := range data1 {
		v := rand.Intn(n * 10)
		data1[i] = v
		data2[i] = v
	}
	
	start := time.Now()
	hybridSortOrdered(data1, 0, len(data1))
	hybridTime := time.Since(start)
	
	start = time.Now()
	sort.Ints(data2)
	stdTime := time.Since(start)
	
	fmt.Printf("Hybrid Sort: %v\n", hybridTime)
	fmt.Printf("std sort:    %v\n", stdTime)
	fmt.Printf("Ratio:       %.2fx\n", float64(hybridTime)/float64(stdTime))
}

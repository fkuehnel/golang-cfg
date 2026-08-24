package hybrid

import (
	"cmp"
	"math/rand"
	"slices"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// safe reference implementations -- the differential oracle
//
// These are the same algorithms written with ordinary branches and indexing.
// The unsafe branchless versions must agree with them exactly, on both the
// returned counts and the resulting array contents.
// ---------------------------------------------------------------------------

func distributeForwardSafe[T cmp.Ordered](pivot T, a, scratch []T) (consumed, nLarger int) {
	n, ss := len(a), len(scratch)
	if n == 0 || ss == 0 {
		return 0, 0
	}
	larger, i := 0, 0
	for i < n {
		x := a[i]
		if !(x < pivot) {
			scratch[ss-1+larger] = x
			larger--
		} else {
			a[i+larger] = x
		}
		i++
		if larger == -ss {
			break
		}
	}
	return i, -larger
}

func distributeBackwardSafe[T cmp.Ordered](pivot T, a, scratch []T) (consumed, nSmaller int) {
	n, ss := len(a), len(scratch)
	if n == 0 || ss == 0 {
		return 0, 0
	}
	smaller, j := 0, n
	for j > 0 {
		j--
		x := a[j]
		if x < pivot {
			scratch[smaller] = x
			smaller++
		} else {
			a[j+smaller] = x
		}
		if smaller == ss {
			break
		}
	}
	return n - j, smaller
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func sameMultiset(a, b []int) bool {
	x := slices.Clone(a)
	y := slices.Clone(b)
	slices.Sort(x)
	slices.Sort(y)
	return slices.Equal(x, y)
}

// distributions that stress different quicksort behaviours
var dists = map[string]func(r *rand.Rand, n int) []int{
	"random": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = r.Int()
		}
		return s
	},
	"sorted": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = i
		}
		return s
	},
	"reverse": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = n - i
		}
		return s
	},
	"allEqual": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = 42
		}
		return s
	},
	"fewUnique": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = r.Intn(3)
		}
		return s
	},
	"twoValues": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = (i & 1)
		}
		return s
	},
	"sawtooth": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = i % 64
		}
		return s
	},
	"organpipe": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			if i < n/2 {
				s[i] = i
			} else {
				s[i] = n - i
			}
		}
		return s
	},
	"nearlySort": func(r *rand.Rand, n int) []int {
		s := make([]int, n)
		for i := range s {
			s[i] = i
		}
		for k := 0; k < n/64+1; k++ {
			if n > 1 {
				i, j := r.Intn(n), r.Intn(n)
				s[i], s[j] = s[j], s[i]
			}
		}
		return s
	},
}

// ---------------------------------------------------------------------------
// distribution primitives
// ---------------------------------------------------------------------------

func TestDistributeForwardMatchesSafe(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 7, 63, 64, 127, 128, 129, 500} {
		for _, ss := range []int{1, 2, 16, 128} {
			for trial := 0; trial < 20; trial++ {
				src := dists["random"](r, n)
				for i := range src {
					src[i] %= 1000
				}
				pivot := 500
				a1, a2 := slices.Clone(src), slices.Clone(src)
				s1, s2 := make([]int, ss), make([]int, ss)

				c1, l1 := distributeForward(pivot, a1, s1)
				c2, l2 := distributeForwardSafe(pivot, a2, s2)

				if c1 != c2 || l1 != l2 {
					t.Fatalf("n=%d ss=%d: counts differ: fast=(%d,%d) safe=(%d,%d)", n, ss, c1, l1, c2, l2)
				}
				if !slices.Equal(a1, a2) {
					t.Fatalf("n=%d ss=%d: array differs\n fast=%v\n safe=%v", n, ss, a1, a2)
				}
				if !slices.Equal(s1, s2) {
					t.Fatalf("n=%d ss=%d: scratch differs\n fast=%v\n safe=%v", n, ss, s1, s2)
				}
			}
		}
	}
}

func TestDistributeBackwardMatchesSafe(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 2, 7, 63, 64, 127, 128, 129, 500} {
		for _, ss := range []int{1, 2, 16, 128} {
			for trial := 0; trial < 20; trial++ {
				src := dists["random"](r, n)
				for i := range src {
					src[i] %= 1000
				}
				pivot := 500
				a1, a2 := slices.Clone(src), slices.Clone(src)
				s1, s2 := make([]int, ss), make([]int, ss)

				c1, m1 := distributeBackward(pivot, a1, s1)
				c2, m2 := distributeBackwardSafe(pivot, a2, s2)

				if c1 != c2 || m1 != m2 {
					t.Fatalf("n=%d ss=%d: counts differ: fast=(%d,%d) safe=(%d,%d)", n, ss, c1, m1, c2, m2)
				}
				if !slices.Equal(a1, a2) {
					t.Fatalf("n=%d ss=%d: array differs\n fast=%v\n safe=%v", n, ss, a1, a2)
				}
				if !slices.Equal(s1, s2) {
					t.Fatalf("n=%d ss=%d: scratch differs\n fast=%v\n safe=%v", n, ss, s1, s2)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// the hybrid partition itself
// ---------------------------------------------------------------------------

func TestHybridPartition(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	// sizes deliberately spanning the ScratchSize boundary and several multiples
	for _, n := range []int{ScratchSize + 1, ScratchSize + 2, 2*ScratchSize - 1, 2 * ScratchSize,
		2*ScratchSize + 1, 3 * ScratchSize, 1000, 5000} {
		for name, gen := range dists {
			for trial := 0; trial < 5; trial++ {
				src := gen(r, n)
				a := slices.Clone(src)
				scratch := make([]int, ScratchSize)
				pivot := medianOfThree(a)

				p := hybridPartition(pivot, a, scratch)

				if p < 0 || p > len(a) {
					t.Fatalf("%s n=%d: partition point %d out of range", name, n, p)
				}
				for i := 0; i < p; i++ {
					if !(a[i] < pivot) {
						t.Fatalf("%s n=%d: a[%d]=%v not < pivot %v", name, n, i, a[i], pivot)
					}
				}
				for i := p; i < len(a); i++ {
					if a[i] < pivot {
						t.Fatalf("%s n=%d: a[%d]=%v < pivot %v but is right of partition", name, n, i, a[i], pivot)
					}
				}
				if !sameMultiset(src, a) {
					t.Fatalf("%s n=%d: partition did not preserve elements", name, n)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// small sort
// ---------------------------------------------------------------------------

func TestBubbleSort2(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	for n := 0; n <= 40; n++ {
		for name, gen := range dists {
			for trial := 0; trial < 10; trial++ {
				src := gen(r, n)
				a := slices.Clone(src)
				bubbleSort2(a)
				if !slices.IsSorted(a) {
					t.Fatalf("%s n=%d: not sorted: %v (from %v)", name, n, a, src)
				}
				if !sameMultiset(src, a) {
					t.Fatalf("%s n=%d: elements not preserved", name, n)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------

func TestSort(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	sizes := []int{0, 1, 2, 3, 5, 15, 16, 17, 63, 127, 128, 129, 255, 256, 257,
		1000, 4096, 10000}
	for _, n := range sizes {
		for name, gen := range dists {
			src := gen(r, n)
			got := slices.Clone(src)
			want := slices.Clone(src)
			slices.Sort(want)

			Sort(got)

			if !slices.Equal(got, want) {
				t.Fatalf("%s n=%d: Sort disagrees with slices.Sort", name, n)
			}
		}
	}
}

func TestSortStrings(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	letters := "abcdefghij"
	for _, n := range []int{0, 1, 17, 200, 1000} {
		src := make([]string, n)
		for i := range src {
			b := make([]byte, 1+r.Intn(4))
			for j := range b {
				b[j] = letters[r.Intn(len(letters))]
			}
			src[i] = string(b)
		}
		got := slices.Clone(src)
		want := slices.Clone(src)
		slices.Sort(want)
		Sort(got)
		if !slices.Equal(got, want) {
			t.Fatalf("strings n=%d: got %v want %v", n, got, want)
		}
	}
}

// Sort must be safe under concurrent use. The previous implementation held its
// scratch buffer in a package-level variable and reallocated it in place, which
// made this test a data race and could silently corrupt results.
func TestSortConcurrent(t *testing.T) {
	const goroutines = 16
	const n = 5000
	var wg sync.WaitGroup
	errs := make([]bool, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(100 + g)))
			for iter := 0; iter < 20; iter++ {
				s := make([]int, n)
				for i := range s {
					s[i] = r.Int()
				}
				want := slices.Clone(s)
				slices.Sort(want)
				Sort(s)
				if !slices.Equal(s, want) {
					errs[g] = true
					return
				}
			}
		}(g)
	}
	wg.Wait()
	for g, bad := range errs {
		if bad {
			t.Fatalf("goroutine %d produced an incorrect sort", g)
		}
	}
}

// ---------------------------------------------------------------------------
// benchmarks
// ---------------------------------------------------------------------------

func benchData(n int, seed int64) []int {
	r := rand.New(rand.NewSource(seed))
	s := make([]int, n)
	for i := range s {
		s[i] = r.Int()
	}
	return s
}

func BenchmarkHybrid(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		src := benchData(n, 7)
		buf := make([]int, n)
		b.Run(itoa(n), func(b *testing.B) {
			b.ReportMetric(0, "ns/op")
			for i := 0; i < b.N; i++ {
				copy(buf, src)
				Sort(buf)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/elem")
		})
	}
}

func BenchmarkSlicesSort(b *testing.B) {
	for _, n := range []int{1000, 100000} {
		src := benchData(n, 7)
		buf := make([]int, n)
		b.Run(itoa(n), func(b *testing.B) {
			b.ReportMetric(0, "ns/op")
			for i := 0; i < b.N; i++ {
				copy(buf, src)
				slices.Sort(buf)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/elem")
		})
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// medianOfThree affects pivot quality, not correctness -- a bad pivot still
// yields a valid partition. So no correctness test can catch a broken median.
// Assert the actual postcondition directly.
func TestMedianOfThree(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	for _, n := range []int{1, 2, 3, 4, 17, 128, 1001} {
		for trial := 0; trial < 200; trial++ {
			a := make([]int, n)
			for i := range a {
				a[i] = r.Intn(50)
			}
			got := medianOfThree(a)
			three := []int{a[0], a[len(a)>>1], a[len(a)-1]}
			slices.Sort(three)
			if got != three[1] {
				t.Fatalf("n=%d: medianOfThree=%d, want median of %v = %d", n, got, three, three[1])
			}
		}
	}
}

func BenchmarkSortSmall(b *testing.B) {
	src := benchData(12, 3)
	buf := make([]int, len(src))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		copy(buf, src)
		Sort(buf)
	}
}

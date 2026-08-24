// clang -O3 --std=c++17 qsort.cc
#include <algorithm>
#include <cstdint>

// Lomuto partition scheme for quicksort:
// https://en.wikipedia.org/wiki/Quicksort#Lomuto_partition_scheme

// This is compiled with more instructions in inner loop then next one
size_t LomutoPartition(uint8_t* arr, size_t lo, size_t hi) {
    auto pivot = arr[hi];
    size_t i = lo;
    for(size_t j = lo; j < hi; j++) {
        if(arr[j] < pivot) {
            std::swap(arr[i], arr[j]);
            i++;
        }
    }
    std::swap(arr[i], arr[hi]);
    return i;
}

void Qsort(uint8_t* arr, size_t lo, size_t hi) {
	if (lo < hi) {
		auto p = LomutoPartition(arr, lo, hi);
		Qsort(arr, lo, p - 1);
		Qsort(arr, p + 1, hi);
	}
}

int main() {
  uint8_t slice[] = {20, 8, 5, 1, 3, 11, 9, 7, 5, 10, 6, 2};
  Qsort(slice, 0, sizeof(slice));
  //std::cout << "\n--- Sorted ---\n\n" << slice << std::end;
}
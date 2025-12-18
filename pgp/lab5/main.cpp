#include <cstdio>
#include <cstdlib>
#include <climits>
#include <algorithm>
#include <chrono>

constexpr int BLOCK_SIZE = 512;

void odd_even_sort_block_cpu(int* a, int l, int r) {
    int len = r - l;
    for (int phase = 0; phase < len; ++phase) {
        if ((phase & 1) == 0) {
            for (int i = l; i + 1 < r; i += 2) {
                if (a[i] > a[i + 1])
                    std::swap(a[i], a[i + 1]);
            }
        } else {
            for (int i = l + 1; i + 1 < r; i += 2) {
                if (a[i] > a[i + 1])
                    std::swap(a[i], a[i + 1]);
            }
        }
    }
}

void bitonic_sort(int* a, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; ++i) {
                int ixj = i ^ j;
                if (ixj > i && ixj < n) {
                    bool asc = ((i & k) == 0);
                    if ((a[i] > a[ixj]) == asc)
                        std::swap(a[i], a[ixj]);
                }
            }
        }
    }
}

int main() {
    int n;
    if (fread(&n, sizeof(int), 1, stdin) != 1)
        return 0;

    if (n <= 0)
        return 0;

    int* h_data = (int*)malloc(sizeof(int) * n);
    if (!h_data)
        return 0;

    fread(h_data, sizeof(int), n, stdin);

    int n2 = 1;
    while (n2 < n) n2 <<= 1;

    int* a = (int*)malloc(sizeof(int) * n2);
    for (int i = 0; i < n; ++i)
        a[i] = h_data[i];
    for (int i = n; i < n2; ++i)
        a[i] = INT_MAX;

    // ================== CPU TIMER START ==================
    auto start = std::chrono::high_resolution_clock::now();

    for (int base = 0; base < n2; base += BLOCK_SIZE) {
        int r = std::min(base + BLOCK_SIZE, n2);
        odd_even_sort_block_cpu(a, base, r);
    }

    bitonic_sort(a, n2);

    auto stop = std::chrono::high_resolution_clock::now();
    // ================== CPU TIMER END ==================

    std::chrono::duration<double, std::milli> elapsed = stop - start;
    fprintf(stderr, "CPU sorting time: %.3f ms\n", elapsed.count());

    fwrite(a, sizeof(int), n, stdout);

    free(a);
    free(h_data);
    return 0;
}

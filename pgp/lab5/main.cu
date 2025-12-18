#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <climits>

#define CSC(call)                                                       \
    do {                                                                \
        cudaError_t res = call;                                         \
        if (res != cudaSuccess) {                                       \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(res));       \
            fflush(stderr);                                             \
            exit(1);                                                    \
        }                                                               \
    } while (0)

constexpr int BLOCK_SIZE = 512;
constexpr int GRID_SIZE = 512;
constexpr int TOTAL_THREADS = GRID_SIZE * BLOCK_SIZE; // 262,144

__global__ void odd_even_sort_block(int* data, int n) {
    __shared__ int sdata[BLOCK_SIZE];

    for (int base = 0; base < n; base += TOTAL_THREADS) {
        int tid = threadIdx.x;
        int gid = base + blockIdx.x * blockDim.x + tid;

        if (gid < n)
            sdata[tid] = data[gid];
        else
            sdata[tid] = INT_MAX;
        __syncthreads();

        for (int phase = 0; phase < BLOCK_SIZE; ++phase) {
            int partner = -1;
            if ((phase & 1) == 0) {
                if ((tid & 1) == 0 && tid + 1 < BLOCK_SIZE) partner = tid + 1;
            } else {
                if ((tid & 1) == 1 && tid + 1 < BLOCK_SIZE) partner = tid + 1;
            }

            if (partner != -1) {
                if (sdata[tid] > sdata[partner]) {
                    int tmp = sdata[tid];
                    sdata[tid] = sdata[partner];
                    sdata[partner] = tmp;
                }
            }
            __syncthreads();
        }

        if (gid < n)
            data[gid] = sdata[tid];

        __syncthreads();
    }
}

__global__ void bitonic_merge_global(int* data, int n, int j, int k) {
    for (int base = 0; base < n; base += TOTAL_THREADS) {
        int idx = base + blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) continue;

        int ixj = idx ^ j;
        if (ixj > idx && ixj < n) {
            bool ascending = ((idx & k) == 0);
            int val1 = data[idx];
            int val2 = data[ixj];

            if ((val1 > val2) == ascending) {
                data[idx] = val2;
                data[ixj] = val1;
            }
        }
    }
}

int main() {
    int n;
    if (fread(&n, sizeof(int), 1, stdin) != 1) return 1;
    if (n <= 0) return 0;

    int* h_data = (int*)malloc(sizeof(int) * (size_t)n);
    if (!h_data) { fprintf(stderr, "ERROR: malloc failed\n"); return 1; }
    if (fread(h_data, sizeof(int), n, stdin) != (size_t)n) { free(h_data); return 1; }

    int n_power2 = 1;
    while (n_power2 < n) n_power2 <<= 1;
    int n2 = n_power2;

    int* h_padded = (int*)malloc(sizeof(int) * (size_t)n2);
    if (!h_padded) { free(h_data); fprintf(stderr, "ERROR: malloc failed\n"); return 1; }
    for (int i = 0; i < n; ++i) h_padded[i] = h_data[i];
    for (int i = n; i < n2; ++i) h_padded[i] = INT_MAX;

    int* d_data;
    CSC(cudaMalloc(&d_data, sizeof(int) * (size_t)n2));
    CSC(cudaMemcpy(d_data, h_padded, sizeof(int) * (size_t)n2, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    odd_even_sort_block<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, n2);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    for (int k = 2; k <= n2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_merge_global<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, n2, j, k);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        }
    }

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));

    CSC(cudaMemcpy(h_data, d_data, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost));

    bool sorted = true;
    for (int i = 1; i < n; ++i) {
        if (h_data[i] < h_data[i-1]) {
            sorted = false;
            fprintf(stderr, "Not sorted at %d: %d < %d\n", i, h_data[i], h_data[i-1]);
            break;
        }
    }
    if (!sorted) { fprintf(stderr, "Array is not sorted!\n"); }

    fwrite(h_data, sizeof(int), n, stdout);
    fflush(stdout);

    float elapsed_ms = 0.0f;
    CSC(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // fprintf(stderr, "GPU sorting time: %.3f ms\n", elapsed_ms);

    CSC(cudaFree(d_data));
    free(h_padded);
    free(h_data);
    return 0;
}

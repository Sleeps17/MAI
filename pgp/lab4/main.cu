#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define CSC(call)                                                       \
    do {                                                                \
        cudaError_t res = call;                                         \
        if (res != cudaSuccess) {                                       \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(res));       \
            fflush(stderr);                                             \
            exit(0);                                                    \
        }                                                               \
    } while (0)


static const double EPS_ZERO = 1e-7;

__global__ void copy_abs_col_kernel(const double* A, double* out_abs, int n, int k) {
    int global_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y)
                  + (threadIdx.y * blockDim.x + threadIdx.x);
    int total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    int len = n - k;
    for (int pos = global_id; pos < len; pos += total_threads) {
        int row = k + pos;
        double v = A[(long long)row * n + k];
        out_abs[pos] = fabs(v);
    }
}


__global__ void swap_rows_kernel(double* A, int n, int r1, int r2) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int col = tx; col < n; col += stride) {
        long long i1 = (long long)r1 * n + col;
        long long i2 = (long long)r2 * n + col;
        double t = A[i1];
        A[i1] = A[i2];
        A[i2] = t;
    }
}

__global__ void compute_factors_kernel(const double* A, double* factors, int n, int k, double pivot) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int start = k + 1;
    int len = n - start;
    for (int pos = tx; pos < len; pos += stride) {
        int i = start + pos;
        double val = A[(long long)i * n + k];
        factors[pos] = val / pivot;
    }
}

__global__ void eliminate_kernel(double* A, const double* factors, int n, int k) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    int start_row = k + 1;
    int start_col = k;
    for (int row = start_row + gy; row < n; row += stride_y) {
        double factor = factors[row - (k + 1)];
        for (int col = start_col + gx; col < n; col += stride_x) {
            long long idx = (long long)row * n + col;
            long long idx_p = (long long)k * n + col;
            double pv = A[idx_p];
            A[idx] -= factor * pv;
        }
        if ((start_col + gx) == start_col) {
        }
    }
}

__global__ void zero_column_kernel(double* A, int n, int k) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int start = k + 1;
    for (int i = start + tx; i < n; i += stride) {
        A[(long long)i * n + k] = 0.0;
    }
}

double* read_matrix(int* n_out) {
    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "ERROR: failed to read n\n");
        return NULL;
    }
    if (n <= 0) {
        fprintf(stderr, "ERROR: invalid n\n");
        return NULL;
    }
    long long total = (long long)n * n;
    double* A = (double*)malloc(sizeof(double) * total);
    if (!A) {
        fprintf(stderr, "ERROR: malloc failed\n");
        return NULL;
    }
    for (long long i = 0; i < total; ++i) {
        if (scanf("%lf", &A[i]) != 1) {
            fprintf(stderr, "ERROR: failed to read matrix element\n");
            free(A);
            return NULL;
        }
    }
    *n_out = n;
    return A;
}

int main() {
    int n;
    double* hA = read_matrix(&n);
    if (!hA) {
        return 0;
    }

    long long total = (long long)n * n;
    double* dA = nullptr;
    CSC(cudaMalloc((void**)&dA, sizeof(double) * total));
    CSC(cudaMemcpy(dA, hA, sizeof(double) * total, cudaMemcpyHostToDevice));

    double* d_abs_col = nullptr;
    CSC(cudaMalloc((void**)&d_abs_col, sizeof(double) * n));

    double* d_factors = nullptr;
    CSC(cudaMalloc((void**)&d_factors, sizeof(double) * (n > 0 ? n - 1 : 1)));

    // Создаем события для измерения времени выполнения
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    dim3 blockCopy( 32, 8 );
    dim3 gridCopy(  64, 8  );

    dim3 blockSwap( 256, 1 );
    dim3 gridSwap(  64, 1 );

    dim3 blockFactors( 256, 1 );
    dim3 gridFactors( 64, 1 );

    dim3 blockElim( 32, 8 );
    dim3 gridElim(  64, 8  );

    dim3 blockZero( 256, 1 );
    dim3 gridZero( 64, 1 );

    double det = 1.0;
    int sign = 1;

    for (int k = 0; k < n; ++k) {
        copy_abs_col_kernel<<<gridCopy, blockCopy>>>(dA, d_abs_col, n, k);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());

        thrust::device_ptr<double> dev_ptr(d_abs_col);
        thrust::device_ptr<double> max_it = thrust::max_element(dev_ptr, dev_ptr + (n - k));
        long long max_pos = (long long)(max_it - dev_ptr); // in [0, n-k-1]
        int pivot_row = (int)(k + max_pos);

        double pivot_val;
        CSC(cudaMemcpy(&pivot_val, dA + (long long)pivot_row * n + k, sizeof(double), cudaMemcpyDeviceToHost));

        if (fabs(pivot_val) <= EPS_ZERO) {
            det = 0.0;
            printf("%.10e\n", 0.0);
            CSC(cudaFree(dA));
            CSC(cudaFree(d_abs_col));
            CSC(cudaFree(d_factors));
            free(hA);
            return 0;
        }

        if (pivot_row != k) {
            swap_rows_kernel<<<gridSwap, blockSwap>>>(dA, n, k, pivot_row);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
            sign = -sign;
        }

        double pivot;
        CSC(cudaMemcpy(&pivot, dA + (long long)k * n + k, sizeof(double), cudaMemcpyDeviceToHost));

        det *= pivot;

        if (k < n - 1) {
            compute_factors_kernel<<<gridFactors, blockFactors>>>(dA, d_factors, n, k, pivot);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());

            eliminate_kernel<<<gridElim, blockElim>>>(dA, d_factors, n, k);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());

            zero_column_kernel<<<gridZero, blockZero>>>(dA, n, k);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        }
    }

    det = det * (double)sign;

    if (fabs(det) <= EPS_ZERO) det = 0.0;

    // printf("%.10e\n", det);

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float ms = 0;
    CSC(cudaEventElapsedTime(&ms, start, stop));
    printf("time: %.6f ms\n", ms);

    CSC(cudaFree(dA));
    CSC(cudaFree(d_abs_col));
    CSC(cudaFree(d_factors));
    free(hA);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static const double EPS_ZERO = 1e-7;

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
            fprintf(stderr, "ERROR: failed to read matrix element %lld\n", i);
            free(A);
            return NULL;
        }
    }
    *n_out = n;
    return A;
}

void swap_rows(double* A, int n, int r1, int r2) {
    if (r1 == r2) return;
    long long offset1 = (long long)r1 * n;
    long long offset2 = (long long)r2 * n;
    for (int j = 0; j < n; ++j) {
        double t = A[offset1 + j];
        A[offset1 + j] = A[offset2 + j];
        A[offset2 + j] = t;
    }
}

int main(void) {
    int n;
    double* A = read_matrix(&n);
    if (!A) return 0;

    // =======================
    //      START TIMER
    // =======================
    struct timespec ts1, ts2;
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    // =======================

    double det = 1.0;
    int sign = 1;

    double* factors = (double*)malloc(sizeof(double) * (n > 0 ? n : 1));
    if (!factors) {
        fprintf(stderr, "ERROR: malloc failed for factors\n");
        free(A);
        return 0;
    }

    for (int k = 0; k < n; ++k) {
        int pivot_row = k;
        double max_abs = fabs(A[(long long)k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double val = fabs(A[(long long)i * n + k]);
            if (val > max_abs) {
                max_abs = val;
                pivot_row = i;
            }
        }

        if (max_abs <= EPS_ZERO) {
            // END TIMER
            clock_gettime(CLOCK_MONOTONIC, &ts2);
            double millis = (ts2.tv_sec - ts1.tv_sec) * 1000.0 +
                            (ts2.tv_nsec - ts1.tv_nsec) / 1e6;

            printf("time: %.3f ms\n", millis);
            printf("%.10e\n", 0.0);

            free(factors);
            free(A);
            return 0;
        }

        if (pivot_row != k) {
            swap_rows(A, n, k, pivot_row);
            sign = -sign;
        }

        double pivot = A[(long long)k * n + k];
        det *= pivot;

        if (k < n - 1) {
            for (int i = k + 1; i < n; ++i) {
                factors[i - (k + 1)] = A[(long long)i * n + k] / pivot;
            }
            for (int i = k + 1; i < n; ++i) {
                double factor = factors[i - (k + 1)];
                long long row_i = (long long)i * n;
                long long row_k = (long long)k * n;
                for (int j = k; j < n; ++j) {
                    A[row_i + j] -= factor * A[row_k + j];
                }
                A[row_i + k] = 0.0;
            }
        }
    }

    det = det * (double)sign;
    if (fabs(det) <= EPS_ZERO) det = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    double millis = (ts2.tv_sec - ts1.tv_sec) * 1000.0 +
                    (ts2.tv_nsec - ts1.tv_nsec) / 1e6;

    printf("time: %.3f ms\n", millis);
    //printf("%.10e\n", det);

    free(factors);
    free(A);
    return 0;
}

   ~/Study/pgp/lab4    master ?6  nvprof ./gpu.out < matrix                                                                                                                                             ✔
==9351== NVPROF is profiling process 9351, command: ./gpu.out
time: 16149.474609 ms
==9351== Profiling application: ./gpu.out
==9351== Profiling result:
           Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   98.11%  15.3709s      4999  3.0748ms  11.519us  9.3889ms  eliminate_kernel(double*, double const *, int, int)
                   0.54%  84.537ms      5000  16.907us  12.413us  26.748us  copy_abs_col_kernel(double const *, double*, int, int)
                   0.41%  64.174ms         1  64.174ms  64.174ms  64.174ms  [CUDA memcpy HtoD]
                   0.23%  35.717ms      4999  7.1440us  2.0800us  26.267us  compute_factors_kernel(double const *, double*, int, int, double)
                   0.22%  35.105ms      4999  7.0220us  1.8880us  28.539us  zero_column_kernel(double*, int, int)
                   0.14%  21.309ms     15000  1.4200us     352ns  28.380us  [CUDA memcpy DtoH]
                   0.12%  18.772ms      4989  3.7620us  2.5920us  28.283us  swap_rows_kernel(double*, int, int, int)
                   0.10%  15.937ms      3720  4.2840us  3.1030us  26.555us  _ZN6thrust23THRUST_200802_SM_610_NS8cuda_cub4core13_kernel_agentINS1_8__reduce11ReduceAgentINS0_12zip_iteratorIN4cuda3std3__45tupleIJNS0_10device_ptrIdEENS0_17counting_iteratorIlNS0_11use_defaultESE_SE_EEEEEEEPNSA_IJdlEEESI_iNS1_9__extrema9arg_max_fIdlNS0_4lessIdEEEEEEJSH_SJ_iN3cub17CUB_200802_SM_61013GridEvenShareIiEENSR_9GridQueueIjEESO_EEEvDpT0_
                   0.08%  12.329ms      3720  3.3140us  2.3030us  26.267us  _ZN6thrust23THRUST_200802_SM_610_NS8cuda_cub4core13_kernel_agentINS1_8__reduce11ReduceAgentIPN4cuda3std3__45tupleIJdlEEESB_SA_iNS1_9__extrema9arg_max_fIdlNS0_4lessIdEEEEEEJSB_SB_iSG_EEEvDpT0_
                   0.03%  4.4314ms      3720  1.1910us  1.0230us  29.275us  _ZN6thrust23THRUST_200802_SM_610_NS8cuda_cub4core13_kernel_agentINS1_8__reduce10DrainAgentIiEEJN3cub17CUB_200802_SM_6109GridQueueIjEEiEEEvDpT0_
                   0.03%  4.0626ms      1280  3.1730us  1.6630us  25.563us  _ZN6thrust23THRUST_200802_SM_610_NS8cuda_cub4core13_kernel_agentINS1_8__reduce11ReduceAgentINS0_12zip_iteratorIN4cuda3std3__45tupleIJNS0_10device_ptrIdEENS0_17counting_iteratorIlNS0_11use_defaultESE_SE_EEEEEEEPNSA_IJdlEEESI_iNS1_9__extrema9arg_max_fIdlNS0_4lessIdEEEEEEJSH_SJ_iSO_EEEvDpT0_
     API calls:   96.70%  15.6603s     24986  626.76us     909ns  9.3935ms  cudaDeviceSynchronize
                   1.05%  170.03ms     10001  17.001us  8.6110us  63.786ms  cudaMemcpy
                   0.79%  127.67ms     37426  3.4110us  2.0270us  712.08us  cudaLaunchKernel
                   0.53%  85.647ms      5003  17.119us  2.0060us  65.742ms  cudaMalloc
                   0.39%  63.562ms      5000  12.712us  9.0500us  605.83us  cudaMemcpyAsync
                   0.25%  40.371ms     10000  4.0370us     564ns  48.119us  cudaStreamSynchronize
                   0.11%  17.263ms      5003  3.4500us  1.6890us  3.2007ms  cudaFree
                   0.07%  11.211ms    199388      56ns      41ns  37.585us  cudaGetLastError
                   0.05%  7.4365ms     39881     186ns     128ns  32.987us  cudaGetDevice
                   0.04%  6.2533ms     29880     209ns     121ns  38.807us  cudaDeviceGetAttribute
                   0.02%  4.0446ms      7440     543ns     253ns  32.077us  cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                   0.01%  1.4252ms     24880      57ns      43ns     939ns  cudaPeekAtLastError
                   0.00%  225.80us       114  1.9800us     152ns  86.708us  cuDeviceGetAttribute
                   0.00%  26.990us         1  26.990us  26.990us  26.990us  cuDeviceGetName
                   0.00%  24.995us         2  12.497us  8.0260us  16.969us  cudaEventRecord
                   0.00%  12.015us         2  6.0070us     269ns  11.746us  cuDeviceGet
                   0.00%  8.9820us         2  4.4910us     505ns  8.4770us  cudaEventCreate
                   0.00%  4.7220us         1  4.7220us  4.7220us  4.7220us  cuDeviceGetPCIBusId
                   0.00%  4.2360us         1  4.2360us  4.2360us  4.2360us  cudaFuncGetAttributes
                   0.00%  3.2660us         1  3.2660us  3.2660us  3.2660us  cudaEventSynchronize
                   0.00%  2.8000us         3     933ns     278ns  2.2030us  cuDeviceGetCount
                   0.00%  1.6730us         1  1.6730us  1.6730us  1.6730us  cudaEventElapsedTime
                   0.00%     886ns         1     886ns     886ns     886ns  cuDeviceTotalMem
                   0.00%     640ns         1     640ns     640ns     640ns  cuModuleGetLoadingMode
                   0.00%     346ns         1     346ns     346ns     346ns  cuDeviceGetUuid
                   0.00%     199ns         1     199ns     199ns     199ns  cudaGetDeviceCount

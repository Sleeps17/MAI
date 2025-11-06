#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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


__global__ void kernel(const double *a, const double *b, double *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += totalThreads) {
        result[i] = fminf(a[i], b[i]);
    }
}

void readArray(double* arr, int n) {
    for (int i = 0; i < n; i++) {
        scanf("%lf", &arr[i]);
    }
}

void writeArray(double* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.10e", arr[i]);
        if (i < n - 1) {
            printf(" ");
        }
    }
    printf("\n");
}

int main() {
    // Считываем количество элементов массивов
    int n;
    scanf("%d", &n);

    // Выделяем память для массивов
    double* a = (double*)malloc(n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    double* result = (double*)malloc(n * sizeof(double));
    if (a == NULL || b == NULL || result == NULL) {
        free(a);
        free(b);
        free(result);
        return 0;
    }

    // Считываем элементы массивов
    readArray(a, n);
    readArray(b, n);

    // Выделяем память для массивов на устройстве
    double *deviceA, *deviceB, *deviceResult;
    CSC(cudaMalloc((void**)&deviceA, n * sizeof(double)));
    CSC(cudaMalloc((void**)&deviceB, n * sizeof(double)));
    CSC(cudaMalloc((void**)&deviceResult, n * sizeof(double)));

    // Копируем массивы на устройство
    CSC(cudaMemcpy(deviceA, a, n * sizeof(double), cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(deviceB, b, n * sizeof(double), cudaMemcpyHostToDevice));


    // Создаем события для измерения времени выполнения
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    // Запускаем ядро
    kernel<<<32, 32>>>(deviceA, deviceB, deviceResult, n);
    CSC(cudaDeviceSynchronize());

    // Измеряем время выполнения
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));

    // Копируем результат с устройства на хост
    CSC(cudaMemcpy(result, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Выводим результат
    // printf("elapsed time: %f ms\n", t);
    writeArray(result, n);

    // Освобождаем память
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));
    CSC(cudaFree(deviceA));
    CSC(cudaFree(deviceB));
    CSC(cudaFree(deviceResult));
    free(a);
    free(b);
    free(result);

    return 0;
}

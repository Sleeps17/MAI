#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

__device__ float grey(uchar4 p) {
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__device__ int gradient(float Gx, float Gy) {
    return min(int(sqrtf(Gx * Gx + Gy * Gy)), 255);
}

__global__ void kernel(cudaTextureObject_t tex, uchar4* out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsetY) {
        for (int x = idx; x < w; x += offsetX) {

            if (x >= w || y >= h)
                return;

            uchar4 z1 = tex2D<uchar4>(tex, x - 1.0f, y - 1.0f);
            uchar4 z2 = tex2D<uchar4>(tex, x, y - 1.0f);
            uchar4 z3 = tex2D<uchar4>(tex, x + 1.0f, y - 1.0f);
            uchar4 z4 = tex2D<uchar4>(tex, x - 1.0f, y);
            uchar4 z5 = tex2D<uchar4>(tex, x + 1.0f, y);
            uchar4 z6 = tex2D<uchar4>(tex, x - 1.0f, y + 1.0f);
            uchar4 z7 = tex2D<uchar4>(tex, x, y + 1.0f);
            uchar4 z8 = tex2D<uchar4>(tex, x + 1.0f, y + 1.0f);

            float w1 = grey(z1), w2 = grey(z2), w3 = grey(z3), w4 = grey(z4);
            float w5 = grey(z5), w6 = grey(z6), w7 = grey(z7), w8 = grey(z8);

            float Gx = w3 + w5 + w5 + w8 - w1 - w4 - w4 - w6;
            float Gy = w6 + w7 + w7 + w8 - w1 - w2 - w2 - w3;

            int grad = gradient(Gx, Gy);
            int off = y * w + x;
            out[off] = make_uchar4(grad, grad, grad, 255);
        }
    }
}


uchar4* readFile(const char* filename, int* w, int* h) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        exit(1);
    }

    fread(w, sizeof(int), 1, file);
    fread(h, sizeof(int), 1, file);

    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * (*w) * (*h));
    fread(data, sizeof(uchar4), (*w) * (*h), file);
    fclose(file);
    return data;
}

void writeFile(const char* filename, uchar4* data, int w, int h) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot create file: %s\n", filename);
        exit(1);
    }
    fwrite(&w, sizeof(int), 1, file);
    fwrite(&h, sizeof(int), 1, file);
    fwrite(data, sizeof(uchar4), w * h, file);
    fclose(file);
}

int main() {
    // Читаем имена входного и выходного файлов из stdin.
    char in_name[500], out_name[500];
    fgets(in_name, 500, stdin); in_name[strlen(in_name)-1] = '\0';
    fgets(out_name, 500, stdin); out_name[strlen(out_name)-1] = '\0';

    // Загружаем входное изображение.
    int w, h;
    uchar4* data = readFile(in_name, &w, &h);

    // Создаём CUDA-массив под изображение и копируем данные туда.
    cudaArray* arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    // Описание ресурса для текстуры.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    // Настройки текстуры
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    // Создаём текстурный объект.
    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    // Выделяем память для выходного изображения на GPU.
    uchar4* dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    // Создаем события для измерения времени выполнения
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));


    // Запуск ядра CUDA с сеткой и блоками.
    dim3 blockDim(64, 64); dim3 gridDim(64, 64);
    kernel<<<gridDim, blockDim>>>(tex, dev_out, w, h);
    CSC(cudaDeviceSynchronize());

    // Измеряем время выполнения
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));

    // Копируем результат обратно на хост.
    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    // Сохраняем обработанное изображение.
    printf("elapsed time: %f ms\n", t);
    writeFile(out_name, data, w, h);

    // Освобождаем ресурсы CUDA и чистим память на хосте.
    free(data);
    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    return 0;
}

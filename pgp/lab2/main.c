#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    unsigned char x, y, z, w;
} uchar4;

static inline float grey(uchar4 p) {
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

static inline int gradient(float Gx, float Gy) {
    int g = (int)(sqrtf(Gx * Gx + Gy * Gy) + 0.5f);
    return g > 255 ? 255 : g;
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
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
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

static inline uchar4 get_pixel(uchar4* img, int w, int h, int x, int y) {
    // clamp координаты
    if (x < 0) x = 0;
    if (x >= w) x = w - 1;
    if (y < 0) y = 0;
    if (y >= h) y = h - 1;
    return img[y * w + x];
}

void sobel_filter_cpu(uchar4* input, uchar4* output, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uchar4 z1 = get_pixel(input, w, h, x - 1, y - 1);
            uchar4 z2 = get_pixel(input, w, h, x,     y - 1);
            uchar4 z3 = get_pixel(input, w, h, x + 1, y - 1);
            uchar4 z4 = get_pixel(input, w, h, x - 1, y);
            uchar4 z5 = get_pixel(input, w, h, x + 1, y);
            uchar4 z6 = get_pixel(input, w, h, x - 1, y + 1);
            uchar4 z7 = get_pixel(input, w, h, x,     y + 1);
            uchar4 z8 = get_pixel(input, w, h, x + 1, y + 1);

            float w1 = grey(z1), w2 = grey(z2), w3 = grey(z3);
            float w4 = grey(z4), w5 = grey(z5);
            float w6 = grey(z6), w7 = grey(z7), w8 = grey(z8);

            float Gx = w3 + w5 + w5 + w8 - w1 - w4 - w4 - w6;
            float Gy = w6 + w7 + w7 + w8 - w1 - w2 - w2 - w3;

            int g = gradient(Gx, Gy);
            uchar4 res = { g, g, g, 255 };
            output[y * w + x] = res;
        }
    }
}

int main() {
    char in_name[500], out_name[500];
    fgets(in_name, 500, stdin); in_name[strcspn(in_name, "\n")] = 0;
    fgets(out_name, 500, stdin); out_name[strcspn(out_name, "\n")] = 0;

    int w, h;
    uchar4* input = readFile(in_name, &w, &h);
    uchar4* output = (uchar4*)malloc(sizeof(uchar4) * w * h);

    if (!output) {
        fprintf(stderr, "Memory allocation failed\n");
        free(input);
        exit(1);
    }

    clock_t start = clock();

    sobel_filter_cpu(input, output, w, h);

    clock_t end = clock();
    float t = (float)(end - start) / CLOCKS_PER_SEC * 1000.0;

    // Вывод результата
    printf("elapsed time: %f ms\n", t);
    writeFile(out_name, output, w, h);

    free(input);
    free(output);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_CLASSES 32

struct Pixel { unsigned char r, g, b, a; }; typedef struct Pixel Pixel;

Pixel* readFile(const char* filename, int* w, int* h) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open input file: %s\n", filename);
        return NULL;
    }
    if (fread(w, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(h, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    long long size = (long long)(*w) * (*h);
    Pixel* data = (Pixel*)malloc(size * sizeof(Pixel));
    if (!data) { fclose(f); return NULL; }
    if (fread(data, sizeof(Pixel), size, f) != (size_t)size) {
        fclose(f); free(data); return NULL;
    }
    fclose(f);
    return data;
}

int writeFile(const char* filename, Pixel* data, int w, int h) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open output file: %s\n", filename); return -1; }
    fwrite(&w, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(data, sizeof(Pixel), (size_t)w * h, f);
    fclose(f);
    return 0;
}

// Функция классификации пикселя на CPU
void classifyPixels(const Pixel* img, Pixel* out, int w, int h, int nc,
                    const float h_avg[MAX_CLASSES * 3], const float h_avg_norm[MAX_CLASSES]) {
    long long total = (long long)w * h;
    for (long long idx = 0; idx < total; ++idx) {
        const Pixel p = img[idx];
        float pr = (float)p.r;
        float pg = (float)p.g;
        float pb = (float)p.b;
        float pnorm = sqrtf(pr*pr + pg*pg + pb*pb);

        int best_class = 0;
        float best_cos = -1.0f - 1e-6f;

        for (int j = 0; j < nc; ++j) {
            float ar = h_avg[j*3 + 0];
            float ag = h_avg[j*3 + 1];
            float ab = h_avg[j*3 + 2];
            float anorm = h_avg_norm[j];

            float dot = pr * ar + pg * ag + pb * ab;
            float cosv;
            if (pnorm == 0.0f || anorm == 0.0f) {
                cosv = -1.0f;
            } else {
                cosv = dot / (pnorm * anorm);
                if (cosv > 1.0f) cosv = 1.0f;
                if (cosv < -1.0f) cosv = -1.0f;
            }

            if (cosv > best_cos) {
                best_cos = cosv;
                best_class = j;
            }
        }

        Pixel outp;
        outp.r = p.r;
        outp.g = p.g;
        outp.b = p.b;
        outp.a = (unsigned char)(best_class & 0xFF);
        out[idx] = outp;
    }
}

int main() {
    // Считываем названия входного и выходного файлов
    char in_line[1024], out_line[1024];
    if (!fgets(in_line, sizeof(in_line), stdin)) { fprintf(stderr, "Failed to read input path\n"); return 1; }
    if (!fgets(out_line, sizeof(out_line), stdin)) { fprintf(stderr, "Failed to read output path\n"); return 1; }
    in_line[strcspn(in_line, "\r\n")] = 0;
    out_line[strcspn(out_line, "\r\n")] = 0;

    // Считываем количество классов
    int nc = 0;
    if (scanf("%d", &nc) != 1) { fprintf(stderr, "Failed to read nc\n"); return 1; }
    if (nc <= 0 || nc > MAX_CLASSES) { fprintf(stderr, "nc out of range (1..%d): %d\n", MAX_CLASSES, nc); return 1; }

    // Вычитываем входной файл
    int w = 0, h = 0;
    Pixel* img = readFile(in_line, &w, &h);
    if (!img) { fprintf(stderr, "Failed to read image file: %s\n", in_line); return 1; }

    float h_avg[MAX_CLASSES * 3];
    float h_avg_norm[MAX_CLASSES];
    for (int j = 0; j < MAX_CLASSES; ++j) {
        h_avg[j*3+0] = h_avg[j*3+1] = h_avg[j*3+2] = 0.0f;
        h_avg_norm[j] = 0.0f;
    }

    for (int j = 0; j < nc; ++j) {
        int npj = 0;
        if (scanf("%d", &npj) != 1) { fprintf(stderr, "Failed to read npj for class %d\n", j); free(img); return 1; }
        if (npj <= 0) {
            h_avg[j*3+0] = h_avg[j*3+1] = h_avg[j*3+2] = 0.0f;
            h_avg_norm[j] = 0.0f;
            continue;
        }
        double sumr = 0.0, sumg = 0.0, sumb = 0.0;
        for (int i = 0; i < npj; ++i) {
            int xi = 0, yi = 0;
            if (scanf("%d %d", &xi, &yi) != 2) { fprintf(stderr, "Failed to read coord for class %d sample %d\n", j, i); free(img); return 1; }
            if (xi < 0) xi = 0;
            if (xi >= w) xi = w - 1;
            if (yi < 0) yi = 0;
            if (yi >= h) yi = h - 1;
            long long idx = (long long)yi * w + xi;
            Pixel p = img[idx];
            sumr += (double)p.r;
            sumg += (double)p.g;
            sumb += (double)p.b;
        }
        float avr = (float)(sumr / npj);
        float avg = (float)(sumg / npj);
        float avb = (float)(sumb / npj);
        h_avg[j*3 + 0] = avr;
        h_avg[j*3 + 1] = avg;
        h_avg[j*3 + 2] = avb;
        h_avg_norm[j] = sqrtf(avr*avr + avg*avg + avb*avb);
    }

    // Выделяем память для выходного изображения
    Pixel* out_img = (Pixel*)malloc(sizeof(Pixel) * (long long)w * h);
    if (!out_img) { fprintf(stderr, "Failed to allocate output image\n"); free(img); return 1; }

    clock_t start = clock();

    // Классификация пикселей на CPU
    classifyPixels(img, out_img, w, h, nc, h_avg, h_avg_norm);

    clock_t end = clock();
    float t = (float)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("elapsed time: %f ms\n", t);
    if (writeFile(out_line, out_img, w, h) != 0) {
        fprintf(stderr, "Failed to write output file\n");
    }

    free(img);
    free(out_img);

    return 0;
}

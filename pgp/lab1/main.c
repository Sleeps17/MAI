#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
        return 1;
    }

    // Считываем элементы массивов
    readArray(a, n);
    readArray(b, n);

    clock_t start = clock();

    // Назодим минимумы элементов массивов
    for (int i = 0; i < n; i++) {
        result[i] = fmin(a[i], b[i]);
    }

    clock_t end = clock();
    float t = (float)(end - start) / CLOCKS_PER_SEC * 1000.0;

    // Вывод результата
    printf("elapsed time: %f ms\n", t);
    // writeArray(result, n);

    // Освобождаем память
    free(a);
    free(b);
    free(result);

    return 0;
}

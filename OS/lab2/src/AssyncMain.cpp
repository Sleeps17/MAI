#include <iostream>
#include <vector>
#include <pthread.h>
#include <cmath>
#include <chrono>

using Matrix = std::vector<std::vector<double>>;

// Consts
const double eps = 1e-2;

// Global varialbles
Matrix result;
double coefficient;
int thread_count, k, matrixHeigth, matrixWide, filterHeigth, filterWide; 


// Структура аргумента тред-функции
struct arg_t{
    Matrix* matrix;
    Matrix* filter;
    int id;
};

// Фунция для считывания матрицы
double ScanMatrix(Matrix& matrix){
    double sum = 0.0;

    for(auto& arr : matrix) {
        for(auto& elem : arr) {
            std::cin >> elem;
            sum += elem;
        }
    }

    return sum;
}

// Функция для печати матрицы
void PrintMatrix(Matrix& matrix) {
    std::cout.precision(3);
    for(const auto& arr : matrix) {
        for(const auto& elem : arr) {
            std::cout << std::fixed << elem << " ";
        }
        std::cout << std::endl;
    }
}

// Функция наложения фильтра на элемента с индексами i j
void ApplyFilter(int height, int wide, arg_t* args) {
    const Matrix& matrix = *args -> matrix, & filter = *args -> filter;

    int hCenter =  filterHeigth >> 1, wCenter = filterWide >> 1;
    double sum = 0.0;

    for(int i = 0; (hCenter + i < filterHeigth) && (height + i < matrixHeigth); ++i) {
        for(int j = 0; (wCenter + j < filterWide) && (wide + j < matrixWide); ++j) {
            sum += matrix[height + i][wide + j] * filter[hCenter + i][wCenter + j];
        } 
    }

    result[height][wide] = sum / coefficient;
}

// Функция наложения фильтра многопоточно
void* ThreadFunction(void* argument) {
    arg_t* args = static_cast<arg_t*>(argument);

    for(int i = 0; i < k; ++i) {
        int startRow = ((matrixHeigth + thread_count - 1) / thread_count) * args -> id;
        int endRow = ((matrixHeigth + thread_count - 1) / thread_count) * (args -> id + 1);

        for(int i = startRow; i < matrixHeigth && i < endRow; ++i) {
            for(int j = 0; j < matrixWide; ++j) {
                ApplyFilter(i, j, args);
            }
        }
        std::cout << "HELLO\n";
        for(int i = startRow; i < endRow; ++i) {
            for(int j = 0; j < matrixWide; ++j) {
                std::cout << "HOW\n";
                (*(args -> matrix))[i][j] = result[i][j];
            }
        }
    }

    std::cout << "HI\n";

    pthread_exit(nullptr);
}


int main(int argc, char** argv) {
    
    // read argument
    if (argc < 2) {
        throw std::logic_error("Не задан ключ запуска программы(кол-во потоков)");
    }
    thread_count = atoi(argv[1]);
    if (thread_count <= 0) {
        throw std::invalid_argument("Кол-во потоков не может быть меньше 1");
    }

    // input
    std::cin >> matrixHeigth >> matrixWide;
    if (matrixHeigth <= 0 || matrixWide <= 0) {
        throw std::invalid_argument("Размеры матрицы не могут быть меньше 1");
    }
    Matrix matrix(matrixHeigth, std::vector<double>(matrixWide));
    result = Matrix(matrixHeigth, std::vector<double>(matrixWide));
    ScanMatrix(matrix);
    Matrix copy = matrix;

    std::cin >> filterHeigth >> filterWide;
    if(filterHeigth <= 0 || filterWide  <= 0) {
        throw std::invalid_argument("Размеры фильтра не могут быть меньше 1");
    }
    if (!(filterHeigth & 1) || !(filterWide & 1)) {
        throw std::invalid_argument("Размеры фильтра должны быть четными числами");
    }
    Matrix filter(filterHeigth, std::vector<double>(filterWide));
    coefficient = ScanMatrix(filter);
    if(coefficient < eps) {
        coefficient = 1.0;
    }

    std::cin >> k;
    if (k < 0) {
        throw std::invalid_argument("Кол-во наложений фильтра должно быть больше 0");
    }

    PrintMatrix(filter);
    PrintMatrix(matrix);

    // make thread data
    arg_t args[thread_count];
    for(int i = 0; i < thread_count; i++) {
        args[i].id = i;
        args[i].matrix = &matrix;
        args[i].filter = &filter;
    }

    // Multithreaded apply
    auto start = std::chrono::steady_clock::now();

    pthread_t tid[thread_count];
    for(int i = 0; i < thread_count; i++) {
        pthread_create(&tid[i], nullptr, ThreadFunction, &args[i]);
    }
    for (int i = 0; i < thread_count; ++i) {
        pthread_join(tid[i], nullptr);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Multithreaded applying: " << duration.count() << " microseconds" << std::endl;

    //std::cout << "Resultation matrix:\n";
    //PrintMatrix(matrix);
}
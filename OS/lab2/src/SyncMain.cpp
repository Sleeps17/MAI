#include<vector>
#include<iostream>
#include<chrono>

using Matrix = std::vector<std::vector<double>>;

// Consts
const double eps = 1e-2;

// Global varialbles
Matrix result;
double coefficient;
int thread_count, k, matrixHeigth, matrixWide, filterHeigth, filterWide; 

// Фунция для считывания матрицы
double ScanMatrix(Matrix& matrix){
    int matrixHeigth = matrix.size();
    int matrixWide = matrix[0].size();
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
    int matrixHeigth = matrix.size();
    int matrixWide = matrix[0].size();
    std::cout.precision(3);
    for(const auto& arr : matrix) {
        for(const auto& elem : arr) {
            std::cout << std::fixed << elem << " ";
        }
        std::cout << std::endl;
    }
}

// Функция наложения матрицы свертки однопоточно
void ApplyFilter(const Matrix& matrix, const Matrix& filter, Matrix& result) {

    const int matrixHeigth = matrix.size(), matrixWide = matrix[0].size();
    const int filterHeigth = filter.size(), filterWide = filter[0].size();
    const int hCenter = filterHeigth >> 1, wCenter = filterWide >> 1;
    double sum = 0.0;

    for(int i = 0; i < matrixHeigth; ++i) {
        for(int j = 0; j < matrixWide; ++j) {

            for(int k = 0; (hCenter + k < filterHeigth) && (i + k < matrixHeigth); ++k) {
                for(int l = 0; (wCenter + l < filterWide) && (j + l < matrixWide); ++l) {
                    sum += matrix[i + k][j + l] * filter[hCenter + k][wCenter + l];
                } 
            }

            result[i][j] = sum / coefficient;

        }
    } 
}

int main() {
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

    // Single-threaded apply
    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < k; ++i) {
        ApplyFilter(copy, filter, result);
        copy = result;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Sigle-threaded applying: " << duration.count() << "microseconds" << std::endl;

}
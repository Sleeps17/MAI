#include <iostream>
#include <vector>
#include <random>

std::vector<std::vector<double>> generateMatrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

int main() {
    int rows = 13;
    int cols = 15;

    std::vector<std::vector<double>> matrix = generateMatrix(rows, cols);

    // Вывод матрицы
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

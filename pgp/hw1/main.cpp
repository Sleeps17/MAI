#include <iostream>
#include <cmath>
#include <iomanip>

const float eps = 1e-9f;

int main() {
    float a, b, c;
    std::cin >> a >> b >> c;

    if (std::fabs(a) < eps && std::fabs(b) < eps && std::fabs(c) >= eps) {
        std::cout << "incorrect" << std::endl;
        return 0;
    }

    if (std::fabs(a) < eps && std::fabs(b) < eps && std::fabs(c) < eps) {
        std::cout << "any" << std::endl;
        return 0;
    }

    if (std::fabs(a) < eps && std::fabs(b) >= eps) {
        std::cout << std::fixed << std::setprecision(6) << -c / b << std::endl;
        return 0;
    }

    float D = b * b - 4.0f * a * c;
    if (D > eps) {
        float x1 = (-b + std::sqrt(D)) / (2.0f * a);
        float x2 = (-b - std::sqrt(D)) / (2.0f * a);
        std::cout << std::fixed << std::setprecision(6) << x1 << " " << x2 << std::endl;
    } else if (std::fabs(D) <= eps) {
        float x = -b / (2.0f * a);
        std::cout << std::fixed << std::setprecision(6) << x << std::endl;
    } else {
        std::cout << "imaginary" << std::endl;
    }

    return 0;
}

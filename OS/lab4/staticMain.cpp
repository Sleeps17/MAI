#include <iostream>
#include "function.hpp"

int main() {
    int type = -1;
    while(true) {
        if(std::cin >> type) {
            if(type == 1) {
                double A, deltaX;
                std::cin >> A >> deltaX;
                double res = derivative(A, deltaX);
                std::cout << res << std::endl;
            } else if(type == 2) {
                double A, B;
                std::cin >> A >> B;
                double res = square(A, B);
                std::cout << res << std::endl;
            } else {
                break;
            }
        } else {
            break;
        }
    }
}
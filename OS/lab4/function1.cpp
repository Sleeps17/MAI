#include "function.hpp"
#include <math.h>

double derivative(double A, double deltaX) {
    return (cos(A + deltaX) - cos(A)) / deltaX;
}

double square(double A, double B) {
    return A*B;
}
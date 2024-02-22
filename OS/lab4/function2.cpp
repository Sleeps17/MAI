#include "function.hpp"
#include <math.h>


double derivative(double A, double deltaX) {
    return (cos(A + deltaX) - cos(A - deltaX))/(2*deltaX);
}

double square(double A, double B) {
    return 0.5*A*B;
}
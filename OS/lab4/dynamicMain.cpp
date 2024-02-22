#include <iostream>
#include <dlfcn.h>

typedef enum {
    First,
    Second
} Context;

typedef const char* Error;

typedef double (*DerivativeFunction)(double, double);
typedef double (*SquareFunction)(double, double);

Context context = First;
DerivativeFunction derivative = nullptr;
SquareFunction square = nullptr;
void* handlers[2] = {nullptr, nullptr};
const char* paths[2] = {"./build/libfunction1.so", "./build/libfunction2.so"};

Error TypeZero() {
    if (context == First) {
        context = Second;
    } else {
        context = First;
    }

    if(!handlers[context]) {
        handlers[context] = dlopen(paths[context], RTLD_LAZY);
        if(!handlers[context]) {
            Error err =  dlerror();
            return err;
        }
    }

    square = (SquareFunction)dlsym(handlers[context], "square");
    if (!square) {
        Error err = dlerror();
        return err;
    }
    derivative = (DerivativeFunction)dlsym(handlers[context], "derivative");
    if (!derivative) {
        Error err = dlerror();
        return err;
    }

    return nullptr;
}

Error TypeOne() {
    if(!handlers[context]) {
        handlers[context] = dlopen(paths[context], RTLD_LAZY);
        if(!handlers[context]) {
            Error err =  dlerror();
            return err;
        }
    }

    if(!derivative) {
        derivative = (DerivativeFunction)dlsym(handlers[context], "derivative");
        if(!derivative) {
            Error err = dlerror();
            return err;
        }
    }


    double A, deltaX;
    std::cin >> A >> deltaX;
    double res = derivative(A, deltaX);
    std::cout << res << std::endl;
    return nullptr;
}

Error TypeTwo() {
    if(!handlers[context]) {
        handlers[context] = dlopen(paths[context], RTLD_LAZY);
        if(!handlers[context]) {
            Error err = dlerror();
            return err;
        }
    }

    if (!square) {
        square = (SquareFunction)dlsym(handlers[context], "square");
        if (!square) {
            Error err = dlerror();
            return err;
        }
    }

    double A, B;
    std::cin >> A >> B;
    double res = square(A, B);
    std::cout << res << std::endl;
    return nullptr;
}

void HandleError(Error err) {
    std::cerr << err << std::endl;
    dlclose(handlers[0]);
    dlclose(handlers[1]);
}


int main() {
    int type = -1;
    while(true) {
        if(std::cin >> type) {
            if(type == 1) {
                Error err = TypeOne();
                if (err != nullptr) {
                    HandleError(err);
                    return 1;
                }
            } else if(type == 2) {
                Error err = TypeTwo();
                if (err != nullptr) {
                    HandleError(err);
                    return 1;
                }
            } else if(type == 0) {
                Error err = TypeZero();
                if (err != nullptr) {
                    HandleError(err);
                    return 1;
                }
            }
            else {
                break;
            }
        } else {
            break;
        }
    }

    dlclose(handlers[0]);
    dlclose(handlers[1]);

    return 0;
}
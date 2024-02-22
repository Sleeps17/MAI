#include <fcntl.h>
#include <iostream>
using namespace std; 

int main(int argc, char *argv[]){

    int fd = open(argv[1], O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
    if (fd == -1) {
        cerr << "Ошибка при попытке открыть файл" << '\n';
        exit(EXIT_FAILURE);
    }

    double res, x;
    char c;
    int i{0};
    while (scanf("%lf%c", &x, &c) > 0) {
        if(i == 0) {
            res = x;
            i++;
        } else {
            if(x == 0) {
                cerr << "Ошибка: попытка деления на ноль" << '\n';
                exit(EXIT_FAILURE);
            } else {
                res /= x;
                if (c == '\n') {
                    dprintf(fd, "%lf\n", res);
                    i = 0;
                }
            }
        }
    }
    return 0;
}
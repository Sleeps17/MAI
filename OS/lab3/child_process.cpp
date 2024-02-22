#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <vector>

#include "shared_memory.hpp"

const double eps = 1e-10;

std::vector<double> numbers;

void GetNumbers(const char* str) {
    numbers.clear();
    char number[20];
    int start = 0;
    for(int i = 0; i < strlen(str); ++i) {
        if(str[i] == ' ' || str[i] == '\n' || str[i] == '\t') {
            number[i - start] = '\0';
            numbers.push_back(std::stod(number));
            start = i + 1;
        } else {
            number[i - start] = str[i];
        }
    }
}


int main(int argc, char *argv[]){

    // Открываем файл для записи ответов
    int fd = open(argv[1], O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
    if(fd == -1) {
        throw std::runtime_error("Ошибка при открытии файла");
    }

    // Открываем сегмент shared memory
    int shmfd = shm_open(sharedMemoryName, O_RDWR, S_IRUSR | S_IWUSR);
    if (shmfd == -1) {
        throw std::runtime_error("Ошибка при открытии shared memory");
    }

    // Отображаем участок памяти в структуру
    SharedMemory* sharedMemory = (SharedMemory*)mmap(NULL, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
    if (sharedMemory == MAP_FAILED) {
        throw std::runtime_error("Ошибка при отображении shared memory");
    }

    std::string s = "";
    while (true) {
        // Ждем пока родитель даст сигнал о том, что можно считывать данные из shared memory
        sem_wait(&sharedMemory->semaphore1);
        // Считываем строку чисел
        while(true) {
            sem_wait(&sharedMemory -> semaphore2);
            s += std::string(sharedMemory -> data);
            sem_post(&sharedMemory -> semaphore3);
            if (std::string(sharedMemory -> data).length() < dataSize - 1) {
                break;
            }
        }

        
        // Если строка, пустая завершаем цикл
        if (s.empty()) {
            break;
        }

        double result;

        // Парсим строку
        GetNumbers(s.c_str());
        // Считаем результат
        for(int i = 0; i < numbers.size(); ++i) {
            if(i == 0) {
                result = numbers[i];
            } else if (numbers[i] < eps && numbers[i] > -eps) {
                throw std::logic_error("Ошибка: попытка деления на 0");
            } else {
                result /= numbers[i];
            }
        }

        s.clear();
        // Печатаем результат в файл
        dprintf(fd, "RESULT: %f\n", result);
    }

    // Отключаем отображение
    munmap(sharedMemory, sizeof(SharedMemory));
    // Закрываем файл
    close(fd);

    return 0;
}

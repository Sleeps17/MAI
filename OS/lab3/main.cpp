#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

#include "shared_memory.hpp"

const char* childProcessName = "./bin/child_process";

int main(){
    // Считываем название файла
    std::string fileName;
    std::cout << "Введите название файла: ";
    std::getline(std::cin, fileName);

    // Открываем сегмент shared memory
    int sharedMemoryfd = shm_open(sharedMemoryName, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);// OPENAT
    if (sharedMemoryfd == -1) {
        throw std::runtime_error("Ошибка при открытии shared memory в дочернем процессе");
    }

    // Изменяем размер сегмента до размера структуры
    if (ftruncate(sharedMemoryfd, sizeof(SharedMemory)) == -1) {
        throw std::runtime_error("Ошибка при изменении размера shared memory");
    }

    // Отображаем участок памяти в структуру
    auto* sharedMemory = (SharedMemory*)mmap(NULL, sizeof(SharedMemory), prots, flags, sharedMemoryfd, 0);
    if (sharedMemory == MAP_FAILED) {
        throw std::runtime_error("Ошибка при отображении shared memory");
    }

    // Инициализируем семафор
    if (sem_init(&sharedMemory->semaphore1, 1, 0) == -1) {
        throw std::runtime_error("Ошибка инициализации семафора");
    }

    if(sem_init(&sharedMemory -> semaphore2, 1, 0) == -1) {
        throw std::runtime_error("Ошибка инициализации семафора");
    }

    if(sem_init(&sharedMemory -> semaphore3, 1, 0) == -1) {
        throw std::runtime_error("Ошибка инициализации семафора");
    }

    // Создаем дочерний процесс
    pid_t pid = fork();
    if (pid == -1) {
        throw std::runtime_error("Ошибка при создании дочернего процесса");
    }

    if (pid == 0) {
        // Запускаем child_process.cpp
        // Передаем имя файла, как аргумент запуска
        execl(childProcessName, childProcessName, fileName.c_str(), NULL);
        throw std::runtime_error("Ошибка при запуске программы из дочернего процесса");
    }

    if (pid > 0) {

        // Считываем  строку чисел
        std::string s;
        while (std::getline(std::cin, s)) {
            s += "\n";

            if (s.length() > dataSize) {
                sem_post(&sharedMemory->semaphore1);
                int len = s.length();
                int start = 0;

                while(true) {
                    // s = 3 1 1 1 3\n
                    //     0123456789
                    std::string substr = s.substr(start, len - start > dataSize - 1 ? dataSize - 1 : len - start);
                    strncpy(sharedMemory -> data, substr.c_str(), sizeof(sharedMemory -> data));

                    sem_post(&sharedMemory -> semaphore2);

                    if(substr.length() < dataSize - 1) {
                        break;
                    } else {
                        start += dataSize - 1;
                    }
                    sem_wait(&sharedMemory -> semaphore3);
                }
            } else {
                // Копируем ее в shared memory
                strncpy(sharedMemory->data, s.c_str(), sizeof(sharedMemory->data));
                sharedMemory->data[sizeof(sharedMemory->data) - 1] = '\0';
                
                // Отправляем сигнал в дочерний процесс, что числа можно считывать
                sem_post(&sharedMemory->semaphore1);
                sem_post(&sharedMemory -> semaphore2);
            }
        }

        // Уничтожаем семафоры
        sem_destroy(&sharedMemory->semaphore1);
        sem_destroy(&sharedMemory -> semaphore2);
        sem_destroy(&sharedMemory -> semaphore3);

        // Отключаем отображение
        munmap(sharedMemory, sizeof(SharedMemory));
        // Освобождаем сегмент
        shm_unlink(sharedMemoryName);
    }

    return 0;
}

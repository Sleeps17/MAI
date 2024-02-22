#ifndef SHARED_MEM_H
#define SHARED_MEM_H

#include<semaphore.h>

const size_t dataSize = 10;
const char* sharedMemoryName = "/my_shared_memory";
const int prots = PROT_READ | PROT_WRITE;
const int flags = MAP_SHARED;

struct SharedMemory {
    char data[dataSize];
    sem_t semaphore1;
    sem_t semaphore2;
    sem_t semaphore3;
};

#endif

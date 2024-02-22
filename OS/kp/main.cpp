#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "freeListsAllocator.hpp"
#include "binaryAllocator.hpp"

using Byte = char;
const size_t allocationSize = 1024*1024;

void TestRandomAllocation(FreeListsAllocator::Allocator& a, BinaryAllocator::Allocator& b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, allocationSize);

    size_t cnt = dis(gen) % 5000 + 5000;
    std::vector<size_t> sizes;
    sizes.reserve(cnt);
    for(int i = 0; i < cnt; ++i) {
        sizes.push_back(dis(gen) % 50);
    }


    auto start = std::chrono::steady_clock::now();
    for(auto& size : sizes) {
        auto ptr = a.allocate(size);
    }

    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "FreeListsAllocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << a.peakAlloced << " bytes" << std::endl;

    start = std::chrono::steady_clock::now();
    //int i = 0;
    for(auto& size : sizes) {
        auto ptr = b.allocate(size);
    }

    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Binary Allocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << b.peakAlloced << " bytes" << std::endl;
}

void TestRandomDeallocation(FreeListsAllocator::Allocator& a, BinaryAllocator::Allocator& b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, allocationSize);

    size_t cnt = dis(gen) % 5000 + 5000;
    std::vector<size_t> sizes;
    sizes.reserve(cnt);
    for(int i = 0; i < cnt; ++i) {
        sizes.push_back(dis(gen) % 50);
    }

    std::vector<Byte*> a_ptrs;
    std::vector<Byte*> b_ptrs;

    auto start = std::chrono::steady_clock::now();
    for(auto& size : sizes) {
        auto ptr = a.allocate(size);

        a_ptrs.push_back(ptr);
    }

    for(auto& ptr : a_ptrs) {
        a.deallocate(ptr);
    }

    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "FreeListsAllocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << a.peakAlloced << " bytes" << std::endl;

    start = std::chrono::steady_clock::now();
    //int i = 0;
    for(auto& size : sizes) {
        //std::cout << i << " " << size << std::endl;
        auto ptr = b.allocate(size);
        //i++;
        b_ptrs.push_back(ptr);
    }

    for(auto& ptr : b_ptrs) {
        b.deallocate(ptr);
    }

    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Binary Allocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << b.peakAlloced << " bytes" << std::endl;
}

void TestOrderedAllocation(FreeListsAllocator::Allocator& a, BinaryAllocator::Allocator& b) {
    std::vector<size_t> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    size_t attemps = 100;

    auto start = std::chrono::steady_clock::now();
    for(auto& size : sizes) {
        for(int i = 0; i < attemps; ++i) {
            auto ptr = a.allocate(size);
        }
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "FreeListsAllocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << a.peakAlloced << " bytes" << std::endl;

    start = std::chrono::steady_clock::now();
    for(auto& size : sizes) {
        for(int i = 0; i < attemps; ++i) {
            auto ptr = b.allocate(size);
        }
    }

    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Binary Allocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << b.peakAlloced << " bytes" << std::endl;
}

void TestOrderedDeallocation(FreeListsAllocator::Allocator& a, BinaryAllocator::Allocator& b) {
    std::vector<size_t> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    size_t attemps = 100;

    std::vector<Byte*> a_ptrs;
    std::vector<Byte*> b_ptrs;

    auto start = std::chrono::steady_clock::now();
    for(auto& size : sizes) {
        for(int i = 0; i < attemps; ++i) {
            auto ptr = a.allocate(size);
            a_ptrs.push_back(ptr);
        }
    }

    for(auto& ptr : a_ptrs) {
        a.deallocate(ptr);
    }
    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "FreeListsAllocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << a.peakAlloced << " bytes" << std::endl;

    start = std::chrono::steady_clock::now();
    for(auto& size : sizes) {
        for(int i = 0; i < attemps; ++i) {
            auto ptr = b.allocate(size);
            b_ptrs.push_back(ptr);
        }
    }

    for(auto& ptr : b_ptrs) {
        b.deallocate(ptr);
    }
    stop = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Binary Allocator:\n";
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Peak memory usage: " << b.peakAlloced << " bytes" << std::endl;
}

int main() {
    FreeListsAllocator::Allocator a1(allocationSize);
    BinaryAllocator::Allocator b1(allocationSize);

    std::cout << "Random Allocation:\n";
    TestRandomAllocation(a1, b1);
    std::cout << std::endl;

    FreeListsAllocator::Allocator a2(allocationSize);
    BinaryAllocator::Allocator b2(allocationSize);

    std::cout << "Random Deallocation:\n";
    TestRandomDeallocation(a2, b2);
    std::cout << std::endl;

    FreeListsAllocator::Allocator a3(allocationSize);
    BinaryAllocator::Allocator b3(allocationSize);

    std::cout << "Ordered Allocation:\n";
    TestOrderedAllocation(a3, b3);
    std::cout << std::endl;


    FreeListsAllocator::Allocator a4(allocationSize);
    BinaryAllocator::Allocator b4(allocationSize);

    std::cout << "Ordered Deallocation:\n";
    TestOrderedDeallocation(a4, b4);
}
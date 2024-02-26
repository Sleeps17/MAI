#include <iostream>
#include "sort.hpp"
#include "container.hpp"

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    MyVector<Data> objects;
    uint16_t key;
    uint64_t value;

    while(std::cin >> key >> value) {
        objects.push_back({key, value});
    }

    objects = CountSort(objects);

    for(auto& object : objects) {
        std::cout << object.key << "\t" << object.value << "\n";
    }
}
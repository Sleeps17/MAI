#pragma once

#include <iomanip>
#include <limits>

#include "container.hpp"

constexpr uint16_t MaxKey = std::numeric_limits<uint16_t>::max();
constexpr uint64_t MaxValue = std::numeric_limits<uint64_t>::max();

struct Data {
    uint16_t key;
    uint64_t value;
};

inline MyVector<Data> CountSort(const MyVector<Data>& objects) {

    MyVector<uint64_t> count(MaxKey + 1, 0);
    MyVector<Data> result(objects.length());

    for (const auto&[key, value] : objects) {
        ++count[key];
    }

    for(int i = 1; i < count.length(); ++i) {
        count[i] += count[i - 1];
    }

    for(int i = objects.length() - 1; i >= 0; --i) {
        result[count[objects[i].key] - 1] = objects[i];
        --count[objects[i].key];
    }

    return result;
}
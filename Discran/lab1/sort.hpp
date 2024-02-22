#pragma once

#include <vector>
#include <iomanip>
#include <limits>

constexpr uint16_t MaxKey = std::numeric_limits<uint16_t>::max();
constexpr uint64_t MaxValue = std::numeric_limits<uint64_t>::max();

struct Data {
    uint16_t key;
    uint64_t value;
};

inline std::vector<Data> CountSort(const std::vector<Data>& objects) {

    std::vector<uint64_t> count(MaxKey + 1, 0);
    std::vector<Data> result(objects.size());

    for (const auto&[key, value] : objects) {
        ++count[key];
    }

    for(int i = 1; i < count.size(); ++i) {
        count[i] += count[i - 1];
    }

    for(int i = objects.size() - 1; i >= 0; --i) {
        result[count[objects[i].key] - 1] = objects[i];
        --count[objects[i].key];
    }

    return result;
}
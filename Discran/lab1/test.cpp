#include <gtest/gtest.h>
#include <random>
#include "sort.hpp"
#include "container.hpp"


bool IsSorted(const MyVector<Data>& objects) {
    for (size_t i = 0; i < objects.length() - 1; ++i) {
        if (objects[i].key > objects[i + 1].key) {
            std::cout << i << " " << i + 1 << "\n";
            return false;
        }
    }
    return true;
}

uint16_t RandomKey() {
    std::random_device rd;
    std::mt19937 gen(rd());
    return static_cast<uint16_t>(std::uniform_int_distribution<uint16_t>(0, MaxKey)(gen));
}

uint64_t RandomValue() {
    std::random_device rd;
    std::mt19937 gen(rd());
    return static_cast<uint64_t>(std::uniform_int_distribution<uint64_t>(0, MaxValue)(gen));
}

TEST(CountSortTest, SortedDataTest) {
    MyVector<Data> objects;

for (int i = 0; i < 10000; ++i) {
        objects.push_back(Data{RandomKey(), RandomValue()});
    }

    auto result = CountSort(objects);

    EXPECT_TRUE(IsSorted(result));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

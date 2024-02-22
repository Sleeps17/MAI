#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace BinaryAllocator {

    using Byte = char;

    size_t Log2(size_t blockSize) {
        return static_cast<size_t>(std::log2(blockSize));
    }

    size_t Pow(size_t base, size_t power) {
        return static_cast<size_t>(std::pow(base, power));
    }

    struct Block {
        size_t offset;
        bool taken;
        Block* next;
    };

    class Allocator {
    private:

        static size_t nextPowerOfTwo(size_t size) {
            return static_cast<size_t>(std::pow(2, std::ceil(std::log2(size))));
        }
        
        static Block* CreateBlock(size_t _offset) {
            auto block = (Block*)calloc(1, sizeof(Block));
            block->offset = _offset;
            block->taken = false;
            block->next = nullptr;
            return block;
        }

        std::pair<size_t, Block*> FindBlock(size_t offset) {
            for(int i = 0; i < freeListsSize; ++i) {
                Block* currBlock = freeLists[i];
                while(currBlock != nullptr) {
                    if (currBlock->offset == offset) {
                        return std::make_pair(i, currBlock);
                    }
                    currBlock = currBlock->next;
                }
            }

            return std::make_pair(0, nullptr);
        }

        bool Empty(size_t listIndex) {
            auto currBlock = freeLists[listIndex];
            bool empty = true;

            while(currBlock != nullptr) {
                if (!currBlock->taken) {
                    empty = false;
                }
                currBlock = currBlock->next;
            }

            return empty;
        }

        Block* Back(size_t listIndex) {
            if (freeLists[listIndex] == nullptr) {
                return nullptr;
            }

            Block* currBlock = freeLists[listIndex];
            Block* goalBlock = nullptr;

            while (currBlock != nullptr) {
                if (!currBlock->taken) {
                    goalBlock = currBlock;
                }
                currBlock = currBlock->next;
            }

            return goalBlock;
        }

        Block* Front(size_t listIndex) {
            if (freeLists[listIndex] == nullptr) {
                return nullptr;
            }

            auto currBlock = freeLists[listIndex];
            while(currBlock != nullptr) {
                if (!currBlock->taken) {
                    return currBlock;
                }
                currBlock = currBlock->next;
            }

            return nullptr;
        }

        void PopBlock(size_t listIndex, Block* block) {
            if (block == nullptr) {
                return;
            }

            Block* currBlock = freeLists[listIndex];
            Block* prevBlock = nullptr;
            while(currBlock != block) {
                prevBlock = currBlock;
                currBlock = currBlock -> next;
            }

            if (prevBlock == nullptr) {
                auto removeBlock = freeLists[listIndex];
                freeLists[listIndex] = block -> next;
                free(removeBlock);
                return;
            }

            prevBlock -> next = block -> next;
            free(block);
        }

        void PushFront(size_t listIndex, Block* block) {
            block -> next = freeLists[listIndex];
            freeLists[listIndex] = block;
        }

        void SplitBlock(size_t listIndex) {
            size_t blockSize = Pow(2, listIndex);
            size_t halfBlockSize = blockSize / 2;

            if (freeLists[listIndex] != nullptr) {
                auto lastBlock = Front(listIndex);
                if (lastBlock == nullptr) {
                    throw std::runtime_error("SPLIT NULLPTR");
                }
                size_t blockOffset = lastBlock->offset;
                PopBlock(listIndex, lastBlock);

                auto firstBlock = CreateBlock(blockOffset);

                auto secondBlock = CreateBlock(blockOffset + halfBlockSize);

                PushFront(listIndex - 1, firstBlock);
                PushFront(listIndex - 1, secondBlock);
            }
        }

        size_t totalSize;
        Byte* memory;
        size_t freeListsSize;
        Block** freeLists;

    public:

        size_t currAlloced{0};
        size_t peakAlloced{0};

        explicit Allocator(size_t _size) : totalSize(nextPowerOfTwo(_size)), freeListsSize(Log2(totalSize) + 1) {
            memory = new Byte[totalSize + freeListsSize];

            freeLists = reinterpret_cast<Block**>(memory);

            auto firstBlock = CreateBlock(freeListsSize * sizeof(Block*));

            size_t index = Log2(totalSize);
            for(int i = 0; i < index; ++i) {
                freeLists[i] = nullptr;
            }
            freeLists[index] = firstBlock;
        }

        Byte* allocate(size_t size) {
            if (size <= 0) {
                return nullptr;
            }
            size_t blockSize = nextPowerOfTwo(size);
            size_t listIndex = Log2(blockSize);

            if (listIndex >= freeListsSize) {
                throw std::runtime_error("END OF MEMORY 1");
            }

            if (Empty(listIndex)) {

                int ind = 1;
                while (listIndex + ind < freeListsSize && Empty(listIndex + ind)) {
                    ind++;
                }

                if (ind == freeListsSize) {
                    throw std::runtime_error("END OF MEMORY 2");
                }

                while(ind > 0) {
                    SplitBlock(listIndex + ind);
                    ind--;
                }
            }

            auto neededBlock = Front(listIndex);
            if (neededBlock == nullptr) {
                throw std::runtime_error("END OF MEMORY 3");
            }

            size_t blockOffset = neededBlock->offset;
            neededBlock->taken = true;

            currAlloced += blockSize;
            peakAlloced = std::max(peakAlloced, currAlloced);

            return memory + blockOffset;
        }

        void deallocate(const Byte* ptr) {
            if (ptr == nullptr) {
                return;
            }

            size_t blockOffset = ptr - memory;
            auto finded = FindBlock(blockOffset);
            size_t listIndex = finded.first;
            Block* block = finded.second;

            if (block == nullptr) {
                throw std::runtime_error("CANNOT FIND BLOCK");
            }

            block->taken = false;

            currAlloced -= Pow(2, listIndex);
        }

        ~Allocator() {
            for(int i = 0; i < freeListsSize; ++i) {
                while(freeLists[i] != nullptr) {
                    PopBlock(i, freeLists[i]);
                }
            }

            delete[] memory;
        }
    };
}


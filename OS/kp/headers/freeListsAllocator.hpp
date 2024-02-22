#pragma once

#include <cmath>
#include <vector>
#include <list>


namespace FreeListsAllocator {

    using Byte = char;

    struct Block {
        size_t size{};
        Byte* data{};
        Block* next{};

    };


    class Allocator {
    private:
        Block* freeBlocks;
        Byte* memory;
        size_t totalSize;

        Block* findBestBlock(size_t size) {
            size_t minDiff = 1e15;
            Block* bestBlock = nullptr;
            Block* currBlock = freeBlocks;
            while(currBlock != nullptr) {
                int currDiff = int(currBlock->size) - int(size);

                if (currDiff >= 0 && currDiff < minDiff) {
                    minDiff = currDiff;
                    bestBlock = currBlock;
                }

                currBlock = currBlock->next;
            }

            return bestBlock;
        }

        void addBlock(size_t len, Byte* ptr) {
            auto blockPtr = reinterpret_cast<Block*>(allocate(sizeof(Block))); //Block*
            if (blockPtr == nullptr) {
                throw std::runtime_error("END OF MEMORY");
            }
            blockPtr->size = len;
            blockPtr->data = ptr;
            blockPtr->next = nullptr;

            if (freeBlocks == nullptr) {
                freeBlocks = blockPtr;
                return;
            }

            auto currBlock = freeBlocks;
            while(currBlock -> next != nullptr) {
                currBlock = currBlock -> next;
            }

            currBlock -> next = blockPtr;
        }

    public:

        size_t currAlloced{0};
        size_t peakAlloced{0};

        explicit Allocator(size_t size): totalSize(size) {
            memory = new Byte[totalSize];
            freeBlocks = reinterpret_cast<Block*>(memory);

            auto sz = totalSize - sizeof(Block) - 8;

            auto ptr = reinterpret_cast<size_t*>(memory + sizeof(Block));
            *ptr = sz;

            freeBlocks->size = sz;
            freeBlocks->data = memory + sizeof(Block) + 8;
            freeBlocks->next = nullptr;
        }

        Byte* allocate(size_t _size) {
            if (_size <= 0) {
                return nullptr;
            }

            if (freeBlocks == nullptr) {
                return nullptr;
            }

            _size += 8;

            Block* bestBlock = findBestBlock(_size);

            if (bestBlock == nullptr) {
                return nullptr;
            }

            auto allocedMemPtr = reinterpret_cast<size_t*>(bestBlock->data + bestBlock->size-_size);
            *allocedMemPtr = _size - 8;

            bestBlock -> size -= _size;
            auto sizePtr = reinterpret_cast<size_t*>(bestBlock->data - 8);
            *sizePtr = bestBlock->size;

            currAlloced += _size;
            peakAlloced = std::max(peakAlloced, currAlloced);

            return bestBlock->data + bestBlock->size + 8;
        }

        void deallocate(Byte* ptr) {
            if (ptr == nullptr) {
                return;
            }

            auto* lenPtr = reinterpret_cast<size_t*>(ptr - 8);
            size_t len = *lenPtr;
            addBlock(len, ptr);

            currAlloced -= len;
        }

        ~Allocator() {
            delete[] memory;
        }
    };
}



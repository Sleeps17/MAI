#pragma once

#include <iomanip>
#include <utility>


class Patricia {
private:
    struct Node;
    Node* root;

    static const size_t bit_count = 5;

    [[nodiscard]] Node* search(const std::string& findKey) const;
    void insert(const std::string& key, const uint64_t& value, const size_t& index);
    [[nodiscard]] Node** triple_search(const std::string& findKey) const;
    void clear_node(Node* node);

public:
    void add(const std::string& key, const uint64_t& value);
    void erase(const std::string& key);
    [[nodiscard]] uint64_t at(const std::string& key) const;
    void clear();

    ~Patricia();
};



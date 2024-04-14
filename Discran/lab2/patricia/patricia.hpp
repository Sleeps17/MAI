#pragma once

#include <iomanip>
#include <utility>


class Patricia {
private:
    struct Node;
    Node* root;
    size_t elements_count;

    static const size_t bit_count = 5;

    [[nodiscard]] Node* search(const std::string& findKey) const;
    void insert(const std::string& key, const uint64_t& value, const size_t& index);
    [[nodiscard]] Node** triple_search(const std::string& findKey) const;
    void clear_node(Node* node);
    void count_elements(Node* root, Node* arr[], int64_t& priority) const;
    void print(std::ostream& os, Node* node, const size_t& prev_index) const;

public:
    void add(const std::string& key, const uint64_t& value);
    void erase(const std::string& key);
    [[nodiscard]] uint64_t at(const std::string& key) const;
    void clear();

    void save(std::ofstream& f) const;
    void load(std::ifstream& f);

    friend std::ostream& operator<<(std::ostream& os, const Patricia& tree);

    ~Patricia();
};



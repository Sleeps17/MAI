#include "patricia.hpp"
#include <cstring>
#include <iostream>

struct Patricia::Node {
    std::string key;
    uint64_t value;
    size_t index;
    Node* left;
    Node* right;

    Node(std::string  key, const uint64_t& value, const size_t& index)
            : key(std::move(key)), value(value), index(index), left(nullptr), right(nullptr) {}

    ~Node()= default;
};

void Patricia::add(const std::string& key, const uint64_t& value) {
    if (root == nullptr) {
        root = new Node(key, value, 0);
        root->left = root;
        return;
    }

    Node *founded_node = search(key);

    if(founded_node->key == key) {
        throw std::runtime_error("key already exists");
    }

    bool run = true;
    size_t char_idx = 0;
    while(run) {
        char founded_key = (founded_node->key.size() > char_idx ? founded_node->key[char_idx] : '\0');
        char input_key   = (key.size() > char_idx ? key[char_idx] : '\0');

        for (size_t i = 0; i < bit_count; ++i) {
            bool found_key_bit = founded_key >> (bit_count - 1 - i) & 1;
            bool input_key_bit = input_key >> (bit_count - 1 - i) & 1;

            if (found_key_bit != input_key_bit) {
                insert(key, value, char_idx * bit_count + i + 1);
                run = false;
                break;
            }
        }
        ++char_idx;
    }
}

void Patricia::erase(const std::string& key) {
    if (!root) {
        return;
    }

    Node** triple = triple_search(key);
    Node* delete_node = triple[0], *owner_node = triple[1], *parent_node = triple[2];

    if (delete_node->key == key) {
        throw std::runtime_error("no such word");
    }

    if (delete_node == root && root->left == root) {
        delete root;
        root = nullptr;
        return;
    }

    if(owner_node == delete_node){
        if(parent_node->right == delete_node) {
            if (delete_node->right == delete_node) {
                parent_node->right = delete_node->left;
            }
            else {
                parent_node->right = delete_node->right;
            }
        }
        else {
            if(delete_node->right == delete_node) {
                parent_node->left = delete_node->left;
            }
            else {
                parent_node->left = delete_node->right;
            }
        }

        delete delete_node;
        return;
    }

    Node** owner_triple = triple_search(owner_node->key);
    Node* owner_owner_node = owner_triple[1];

    delete_node->key = owner_node->key;
    delete_node->value = owner_node->value;

    if(owner_owner_node == owner_node){
        if(parent_node->right == owner_node) {
            parent_node->right = delete_node;
        }
        else {
            parent_node->left = delete_node;
        }
    }
    else{
        if(parent_node->right == owner_node) {
            if(owner_node->right == delete_node) {
                parent_node->right = owner_node->left;
            }
            else {
                parent_node->right = owner_node->right;
            }
        }
        else {
            if(owner_node->right == delete_node) {
                parent_node->left = owner_node->left;
            }
            else {
                parent_node->left = owner_node->right;
            }
        }

        if(owner_owner_node->right == owner_node) {
            owner_owner_node->right = delete_node;
        }
        else {
            owner_owner_node->left = delete_node;
        }
    }

    delete triple;
    delete owner_triple;
    delete owner_node;
}

uint64_t Patricia::at(const std::string& key) const {
    if (!root) {
        throw std::runtime_error("tree is empty");
    }

    Node* node = search(key);
    if (node->key == key) {
        return node->value;
    }

    throw std::runtime_error("key not fount");
}

void Patricia::clear() {
    if (!root) {
        return;
    }

    if (root != root->left) {
        clear_node(root->left);
    }

    delete root;
    root = nullptr;
}

Patricia::~Patricia() {
    clear();
}

Patricia::Node* Patricia::search(const std::string& findKey) const {
    Node* curr_node = root->left, *prev_node = root;

    while(curr_node->index > prev_node->index) {
        size_t char_index = (curr_node->index - 1) / bit_count;

        if (char_index >= findKey.size()) {
            prev_node = curr_node;
            curr_node = curr_node -> left;
            continue;
        }

        char curr_char = findKey[char_index];
        size_t offset = (bit_count - 1 - ((curr_node->index - 1) % bit_count));
        bool curr_bit = (curr_char >> offset) & 1;

        prev_node = curr_node;
        curr_node = curr_bit ? curr_node->right : curr_node->left;
    }

    return curr_node;
}

void Patricia::insert(const std::string& key, const uint64_t &value, const size_t &index) {
    Node* curr_node = root->left, *prev_node = root;

    while(curr_node->index > prev_node->index) {
        if (curr_node->index > index) {
            break;
        }

        size_t char_idx = (curr_node->index - 1)/bit_count;
        if (char_idx >= key.length()) {
            prev_node = curr_node;
            curr_node = curr_node->left;
            continue;
        }

        char curr_char = key[char_idx];
        size_t offset = (bit_count - 1 - ((curr_node->index - 1) % bit_count));
        bool curr_bit = (curr_char >> offset) & 1;

        prev_node = curr_node;
        curr_node = curr_bit ? curr_node->right : curr_node->left;
    }

    char char_from_key = key[(index - 1) / bit_count];
    bool get_bit = char_from_key >> (bit_count - 1 - (index - 1) % bit_count) & 1;

    Node* new_node = new Node(key, value, index);

    if(prev_node->left == curr_node) {
        prev_node->left = new_node;
    }
    else {
        prev_node->right = new_node;
    }

    if (get_bit) {
        new_node->right = new_node;
        new_node->left = curr_node;
    } else {
        new_node->left = new_node;
        new_node->right = curr_node;
    }
}

Patricia::Node** Patricia::triple_search(const std::string& findKey) const {
    Node *curr_node = root->left, *prev_node = root, *prev_prev_node = root;

    while(curr_node->index > prev_node->index) {
        size_t char_idx = (curr_node->index - 1) / bit_count;

        if (char_idx >= findKey.length()) {
            prev_prev_node = prev_node;
            prev_node = curr_node;
            curr_node = curr_node->left;
            continue;
        }

        char curr_char = findKey[char_idx];
        size_t offset = (bit_count - 1 - ((curr_node->index - 1) % bit_count));
        bool curr_bit = (curr_char >> offset) & 1;

        prev_prev_node = prev_node;
        prev_node = curr_node;
        if (curr_bit) {
            curr_node = curr_node->right;
        } else {
            curr_node = curr_node->left;
        }
    }

    Node** result = new Node*[3];
    result[0] = curr_node;
    result[1] = prev_node;
    result[2] = prev_prev_node;

    return result;
}

void Patricia::clear_node(Patricia::Node *node) {
    if(node->left->index > node->index) {
        clear_node(node->left);
    }
    if(node->right->index > node->index) {
        clear_node(node->right);
    }

    delete node;
}



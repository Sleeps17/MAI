#include <iostream>
#include <fstream>
#include <utility>
#include <cstring>
#include <cstdint>

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

struct Patricia::Node {
    std::string key{};
    uint64_t value{};
    size_t index{};
    int64_t priority{};
    Node* left{};
    Node* right{};

    Node() = default;

    Node(std::string  key, const uint64_t& value, const size_t& index)
            : key(std::move(key)), value(value), index(index), left(nullptr), right(nullptr) {
        priority = -1;
    }

    Node(std::string key, const uint64_t& value, const size_t& index, Node* left, Node* right)
            : key(std::move(key)), value(value), index(index), left(left), right(right) {
        priority = -1;
    }

    ~Node()= default;
};

void Patricia::add(const std::string& key, const uint64_t& value) {
    if (root == nullptr) {
        root = new Node(key, value, 0);
        root->left = root;
        elements_count++;
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

    elements_count++;
}

void Patricia::erase(const std::string& key) {
    if (!root) {
        throw std::runtime_error("no such word");
    }

    Node** triple = triple_search(key);
    Node* delete_node = triple[0], *owner_node = triple[1], *parent_node = triple[2];

    if (delete_node->key != key) {
        throw std::runtime_error("no such word");
    }

    if (delete_node == root && root->left == root) {
        delete root;
        root = nullptr;
        elements_count--;
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

    elements_count--;
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
    elements_count = 0;
}

Patricia::~Patricia() {
    clear();
}

void Patricia::save(std::ofstream& f) const {
    f.write(reinterpret_cast<const char*>(&elements_count), sizeof(size_t));

    if (elements_count > 0) {
        Node* arr[elements_count];
        int64_t priority = 0;
        count_elements(root, arr, priority);

        for (size_t i = 0; i < elements_count; ++i) {
            f.write(reinterpret_cast<const char*>(&arr[i]->value), sizeof(int64_t));
            f.write(reinterpret_cast<const char*>(&arr[i]->index), sizeof(size_t));
            size_t length = 0;
            if (arr[i] != nullptr) {
                length = arr[i]->key.length();
            }

            f.write(reinterpret_cast<const char*>(&length), sizeof(size_t));
            f.write(arr[i]->key.c_str(), long(length*sizeof(char)));
            int64_t left_priority = -1;
            if (arr[i] != nullptr && arr[i]->left != nullptr) {
                left_priority = arr[i]->left->priority;
            }
            f.write(reinterpret_cast<const char*>(&left_priority), sizeof(size_t));
            int64_t right_priority = -1;
            if (arr[i] != nullptr && arr[i]->right != nullptr) {
                right_priority = arr[i]->right->priority;
            }
            f.write(reinterpret_cast<const char*>(&right_priority), sizeof(size_t));
        }
    }
}

void Patricia::load(std::ifstream& f) {
    this->clear();

    size_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(size_t));

    if (count == 0) {
        return;
    }

    elements_count = count;

    root = new Node();
    Node* arr[elements_count];
    arr[0] = root;
    for (size_t i = 1; i < elements_count; ++i) {
        arr[i] = new Node();
    }

    std::string key;
    size_t value, index, length;
    int64_t left_priority = -1, right_priority = -1;
    for(size_t i = 0; i < elements_count; ++i) {
        f.read(reinterpret_cast<char*>(&value), sizeof(int64_t));
        f.read(reinterpret_cast<char*>(&index), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&length), sizeof(size_t));
        key.resize(length);
        f.read(const_cast<char*>(key.data()), long(length*sizeof(char)));
        f.read(reinterpret_cast<char*>(&left_priority), sizeof(size_t));
        f.read(reinterpret_cast<char*>(&right_priority), sizeof(size_t));
        *arr[i] = Node(key, value, index, left_priority >= 0 ? arr[left_priority] : nullptr, right_priority >= 0 ? arr[right_priority] : nullptr);
    }
}

std::ostream &operator<<(std::ostream &os, const Patricia &tree) {
    if (tree.root == nullptr) {
        return os;
    }

    os << '\t'  << "[ " << tree.root->index << ' ' << tree.root->key << ' ' << tree.root->value << " ]" << '\n';
    tree.print(os, tree.root->left, tree.root->index);

    return os;
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
    if (node == nullptr) {
        return;
    }

    if(node->left->index > node->index) {
        clear_node(node->left);
    }
    if(node->right->index > node->index) {
        clear_node(node->right);
    }

    delete node;
}

void Patricia::count_elements(Patricia::Node* _root, Patricia::Node **arr, int64_t &priority) const {
    if (_root == nullptr) {
        return;
    }

    _root->priority = priority;
    arr[priority++] = _root;

    if (_root->left != nullptr && _root->left->index > _root->index) {
        count_elements(_root->left, arr, priority);
    }

    if (_root->right != nullptr && _root->right->index > _root->index) {
        count_elements(_root->right, arr, priority);
    }
}

void Patricia::print(std::ostream& os, Patricia::Node *node, const size_t &prev_index) const {
    if(node->index <= prev_index)
        return;

    os << '\t'  << "[ " << node->index << ' ' << node->key << ' ' << node->value << " ]" << '\n';
    print(os, node->left, node->index);
    print(os, node->right, node->index);
}

void to_lower_case(std::string& str) {
    for(int i = 0; i < str.length(); i++) {
        str[i] = (char)tolower(str[i]);
    }
}

int main() {
    std::string input;
    Patricia p{};

    while(std::cin >> input) {
        bool catched = false;
        if (input == "+") {
            uint64_t value;
            std::string key;
            // Read key and value
            std::cin >> key >> value;
            // Transform key to lower case
            to_lower_case(key);
            // Add word
            try {
                p.add(key, value);
            } catch(std::runtime_error& ex) {
                catched = true;
                std::cout << "Exist" << '\n';
            }

            if (!catched) {
                std::cout << "OK" << '\n';
            }
        } else if (input == "-") {
            // Read key
            std::string key;
            std::cin >> key;
            // Transform key to lower case
            to_lower_case(key);
            // Delete word
            try {
                p.erase(key);
            } catch (std::runtime_error& ex) {
                catched = true;
                std::cout << "NoSuchWord" << '\n';
            }

            if (!catched) {
                std::cout << "OK" << '\n';
            }
        } else if (input == "!") {
            std::string cmd, path;
            std::cin >> cmd >> path;
            if (cmd == "Save") {
                std::ofstream f;
                f.open(path, std::ios::trunc | std::ios::out | std::ios::binary);
                p.save(f);
                std::cout << "OK" << '\n';
            } else {
                std::ifstream f;
                f.open(path, std::ios::binary | std::ios::in);
                p.load(f);
                std::cout << "OK" << '\n';
            }
        } else {
            uint64_t value;
            to_lower_case(input);
            try {
                value = p.at(input);
            }
            catch(std::runtime_error& ex) {
                catched = true;
                std::cout << "NoSuchWord" << '\n';
            }

            if (!catched) {
                std::cout << "OK: " << value << '\n';
            }
        }
//        std::cout << p << std::endl;
    }
}
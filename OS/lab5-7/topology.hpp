#ifndef TOPOLOGY_H
#define TOPOLOGY_H

typedef const char* Error;

template <typename T>
class Topology {
private:
    struct Node {
        T data;
        Node* left;
        Node* right;

        explicit Node(const T& value) : data(value), left(nullptr), right(nullptr) {}
    };

    Node* root;

public:
    Topology() : root(nullptr) {}
    ~Topology() {
        destroyTree(root);
    }

    Error insert(const T& value) {
        bool have = searchNode(root, value);
        if (have) {
            return "Error: Already exists";
        }
        root = insertNode(root, value);
        return nullptr;
    }

    bool search(const T& value) {
        return searchNode(root, value);
    }

    Error remove(const T& value) {
        bool have = searchNode(root, value);
        if(!have) {
            return "Error: id not found";
        }
        root = deleteNode(root, value);
        return nullptr;
    }

    std::vector<T> findPathToNode(const T& value) {
        std::vector<T> path;
        findPath(root, value, path);
        return path;
    }

    int maxDepth() {
        return privateMaxDepth(root);
    }

    std::vector<T> getNodesInSubtree(const T& value) {
        Node* node = findNode(root, value);
        if (node == nullptr) {
            return {};
        }
        std::vector<T> nodes;
        getNodes(node, nodes);
        return nodes;
    }

private:
    Node* insertNode(Node* node, const T& value) {
        if (node == nullptr) {
            return new Node(value);
        }

        if (value < node->data) {
            node->left = insertNode(node->left, value);
        } else if (value > node->data) {
            node->right = insertNode(node->right, value);
        }

        return node;
    }

    bool searchNode(Node* node, const T& value) {
        if (node == nullptr) {
            return false;
        }

        if (value == node->data) {
            return true;
        } else if (value < node->data) {
            return searchNode(node->left, value);
        } else {
            return searchNode(node->right, value);
        }
    }

    void destroyTree(Node* node) {
        if (node == nullptr) {
            return;
        }

        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }

    Node* findSuccessor(Node* node) {
        Node* current = node;
        while (current && current->left != nullptr) {
            current = current->left;
        }
        return current;
    }

    Node* deleteNode(Node* node, const T& value) {
        if (node == nullptr) {
            return nullptr;
        }

        if (value < node->data) {
            node->left = deleteNode(node->left, value);
        } else if (value > node->data) {
            node->right = deleteNode(node->right, value);
        } else {
            if (node->left == nullptr && node->right == nullptr) {
                delete node;
                return nullptr;
            } else if (node->left == nullptr) {
                Node* temp = node->right;
                delete node;
                return temp;
            } else if (node->right == nullptr) {
                Node* temp = node->left;
                delete node;
                return temp;
            } else {
                Node* successor = findSuccessor(node->right);
                node->data = successor->data;
                node->right = deleteNode(node->right, successor->data);
            }
        }

        return node;
    }

    bool findPath(Node* node, const T& value, std::vector<T>& path) {
        if (node == nullptr) {
            return false;
        }

        if (value == node->data) {
            path.push_back(node->data);
            return true;
        }
        else if (value < node->data) {
            if (findPath(node->left, value, path)) {
                path.push_back(node->data);
                return true;
            }
        }
        else {
            if (findPath(node->right, value, path)) {
                path.push_back(node->data);
                return true;
            }
        }
        return false;
    }

    int privateMaxDepth(Node* node) {
        if (node == nullptr) {
            return 0;
        }

        int leftDepth = privateMaxDepth(node->left);
        int rightDepth = privateMaxDepth(node->right);

        return std::max(leftDepth, rightDepth) + 1;
    }

    Node* findNode(Node* node, const T& value) {
        if (node == nullptr || node->data == value) {
            return node;
        }

        if (value < node->data) {
            return findNode(node->left, value);
        } else {
            return findNode(node->right, value);
        }
    }

    void getNodes(Node* node, std::vector<T>& nodes) {
        if (node == nullptr) {
            return;
        }

        getNodes(node->left, nodes);
        nodes.push_back(node->data);
        getNodes(node->right, nodes);
    }
};

#endif //TOPOLOGY_H

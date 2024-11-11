#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>

std::vector<int> topological_sort(const int n, std::vector<std::pair<int, int>> &constraints) {
    std::vector<std::vector<int>> graph(n + 1);
    std::vector in_degree(n + 1, 0);

    for (const auto &[a, b] : constraints) {
        graph[a].push_back(b);
        in_degree[b]++;
    }

    std::queue<int> q;
    for (int i = 1; i <= n; ++i) {
        if (in_degree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : graph[node]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if (result.size() != n) {
        return {-1};
    }

    return result;
}

int main() {
    int n, m;
    std::cin >> n >> m;
    std::vector<std::pair<int, int>> constraints(m);
    for (int i = 0; i < m; ++i) {
        std::cin >> constraints[i].first >> constraints[i].second;
    }

    std::vector<int> sorted_order = topological_sort(n, constraints);
    if (sorted_order.size() == 1 && sorted_order[0] == -1) {
        std::cout << -1 << "\n";
    } else {
        for (const int num : sorted_order) {
            std::cout << num << " ";
        }
        std::cout << "\n";
    }
}

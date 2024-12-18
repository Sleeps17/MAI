#include <vector>
#include <algorithm>
#include <iostream>
#include <queue>
#include <cstdint>
#include <limits>

const int noPrevVertex = -1;

using Graph = std::vector<std::vector<int64_t>>;

Graph make_graph(uint vertices) {
    return std::vector<std::vector<int64_t>>(vertices, std::vector<int64_t>(vertices));
}

void bfs(Graph const& graph, Graph const& flow, uint from, std::vector<uint> & prev) {
    std::queue<uint> q;
    q.push(from);
    prev[from] = from;

    while (not q.empty()) {
        from = q.front();
        q.pop();

        for (size_t idx = 0; idx < graph[from].size(); ++idx) {
            uint to = idx;
            if (prev[to] == noPrevVertex and flow[from][to] < graph[from][to]) {
                prev[to] = from;
                q.push(to);
            }
        }
    }
}

int main() {
    uint n, m;
    std::cin >> n >> m;
    auto graph = make_graph(n);
    auto flow = make_graph(n);

    for (size_t idx = 0; idx < m; ++idx) {
        uint from, to;
        int64_t weight;
        std::cin >> from >> to >> weight;
        graph[from-1][to-1] += weight;
    }

    uint start = 1, finish = n;
    uint64_t ans = 0;

    while (true) {
        std::vector<uint> prev(n, noPrevVertex);
        bfs(graph, flow, start-1, prev);
        if (prev[finish - 1] == noPrevVertex) {
            break;
        }

        std::vector<uint> path;
        uint last = finish - 1;
        while (prev[last] != last) {
            path.push_back(last);
            last = prev[last];
        }
        path.push_back(last);
        std::reverse(path.begin(), path.end());

        int64_t min_flow = std::numeric_limits<int64_t>::max();
        for (size_t idx = 1; idx < path.size(); ++idx) {
            uint from = path[idx - 1], to = path[idx];
            min_flow = std::min(graph[from][to] - flow[from][to], min_flow);
        }

        for (size_t i = 1; i < path.size(); ++i) {
            uint from = path[i - 1], to = path[i];
            flow[from][to] += min_flow;
            flow[to][from] -= min_flow;
        }
        ans += min_flow;
    }
    std::cout << ans << std::endl;
}
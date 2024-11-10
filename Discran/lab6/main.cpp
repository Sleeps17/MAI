#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>

const std::vector<std::pair<int, int>> moves = {{-1, 1}, {0, 1}, {1, 1}};

bool can_move(int x, int y, int x_move, int y_move, int n, int m) {
    return x + x_move >= 0 && x + x_move < m && y + y_move >= 0 && y + y_move < n;
}

int main() {
    int n, m;
    std::cin >> n >> m;
    std::vector A (n, std::vector<int64_t>(m));

    for(int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cin >> A[i][j];
        }
    }

    std::vector prev_col(n, std::vector<std::pair<int, int>>(m));

    for(int y = n - 2; y >= 0; --y) {
        for (int x = 0; x < m; ++x) {

            int64_t min_prev_a = std::numeric_limits<int64_t>::max();
            std::pair<int, int> col;

            for (const auto &[x_move, y_move] : moves) {
                if (!can_move(x, y, x_move, y_move, n, m)) {
                    continue;
                }

                if (A[y+y_move][x+x_move] < min_prev_a) {
                    min_prev_a = A[y+y_move][x+x_move];
                    col = {y+y_move, x+x_move};
                }
            }

            prev_col[y][x] = {col.first, col.second};
            A[y][x] += min_prev_a;
        }
    }

    int64_t minimum = A[0][0];
    int idx = 0;
    for(int i = 1; i < m; ++i) {
        if (A[0][i] <= minimum) {
            minimum = A[0][i];
            idx = i;
        }
    }

    std::cout << minimum << "\n";

    int prev_x = idx, prev_y = 0;
    int counter = 0;
    while(counter < n) {
        std::cout << "(" << prev_y + 1 << "," << prev_x + 1 << ")";
        if (counter != n-1) {
            std::cout << " ";
        }
        int new_prev_y = prev_col[prev_y][prev_x].first, new_prev_x = prev_col[prev_y][prev_x].second;
        prev_y = new_prev_y, prev_x = new_prev_x;
        counter++;
    }
}
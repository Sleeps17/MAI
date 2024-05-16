#include <sstream>
#include "iostream"
#include "vector"
#include "utility"

const int MAX = 256;

void to_lower_case(std::string& str) {
    for(size_t i = 0; i < str.length(); i++) {
        str[i] = static_cast<char>(tolower(str[i]));
    }
}

void badCharHeuristic(const std::string& str, int size, int bad_char[MAX]) {
    for (int i = 0; i < MAX; i++) {
        bad_char[i] = size;
    }
    for (int i = 0; i < size - 1; i++) {
        bad_char[static_cast<unsigned char>(str[i])] = size - i - 1;
    }
}

std::vector<uint> search(const std::string& txt, const std::string& pat) {
    std::vector<uint> answer;

    int n = static_cast<int>(txt.length());
    int m = static_cast<int>(pat.length());

    int bad_char[MAX];
    badCharHeuristic(pat, m, bad_char);

    int s = 0;
    while (s <= (n - m)) {
        int j = m - 1;
        while (j >= 0 && pat[j] == txt[s + j]) {
            j--;
        }
        if (j < 0) {
            answer.push_back(s);
            s += m;
        } else {
            s += bad_char[static_cast<unsigned char>(txt[s + j])] - m + j + 1 >= 1 ? bad_char[static_cast<unsigned char>(txt[s + j])] - m + j + 1 : 1;
        }
    }

    return answer;
}

std::vector<int> findPattern(const std::string& text, const std::string& pattern) {
    std::vector<int> positions;

    size_t pos = text.find(pattern, 0);
    while (pos != std::string::npos) {
        positions.push_back(pos);
        pos = text.find(pattern, pos + 1);
    }

    return positions;
}

int main() {
    std::string pattern;
    std::getline(std::cin, pattern);
    to_lower_case(pattern);

    std::vector<uint> indexes;
    std::vector<std::pair<int, int>> positions;

    std::string text, line;
    uint idx = 0, line_idx = 1, word_idx = 1;
    while (std::getline(std::cin, line)){
        std::stringstream iss(line);

        std::string word;
        while(iss >> word) {
            indexes.push_back(idx);
            positions.emplace_back(line_idx, word_idx);

            idx += word.length() + 1;
            word_idx++;

            text += word + " ";
        }

        line_idx++;
        word_idx = 1;
    }

    text = text.substr(0, text.length() - 1);

    to_lower_case(text);

    auto answer = search(text, pattern);

    for (auto& _idx : answer) {
        if ((_idx != 0 && text[_idx-1] != ' ') || (_idx + pattern.length() != text.length() && text[_idx + pattern.length()] != ' ')) {
            continue;
        }
        auto it = std::lower_bound(indexes.begin(), indexes.end(), _idx);
        if (*it != _idx || it == indexes.end()) {
            continue;
        }
        auto pos = std::distance(indexes.begin(), it);

        std::cout << positions[pos].first << ", " << positions[pos].second << "\n";
    }

    return 0;
}
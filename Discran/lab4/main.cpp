#include "unordered_map"
#include "sstream"
#include "iostream"
#include "vector"
#include "utility"

struct Triple {
    std::string word;
    uint line_idx;
    uint word_idx;
};

using Text = std::vector<Triple>;
using Pattern = std::vector<std::string>;

void to_lower_case(std::string& str) {
    for(size_t i = 0; i < str.length(); i++) {
        str[i] = static_cast<char>(tolower(str[i]));
    }
}

int bad_symbol_heuristic_shift(std::unordered_map<std::string, int> &table, int i, const std::string &s) {
    return table.count(s) == 0 ? i + 1 : (i - table[s] > 0 ? i - table[s] : 1);
}

std::vector<std::pair<uint, uint>> search(const Text& text, const Pattern& pattern) {
    std::vector<std::pair<uint, uint>> answer;

    uint pattern_size = pattern.size();
    uint text_size = text.size();

    std::unordered_map<std::string, int> stop_table;
    for(size_t pattern_iterator = 0; pattern_iterator < pattern_size; pattern_iterator++) {
        stop_table[pattern[pattern_iterator]] = static_cast<int>(pattern_iterator);
    }

    int i = static_cast<int>(pattern_size - 1);
    while(i >= 0 && i < static_cast<int>(text_size)) {
        int text_idx = i;
        int pattern_idx = static_cast<int>(pattern_size - 1);

        while(text[text_idx].word == pattern[pattern_idx]) {
            if(pattern_idx == 0) {
                answer.emplace_back(text[text_idx].line_idx, text[text_idx].word_idx);
                break;
            }

            text_idx--;
            pattern_idx--;
        }

        i += std::max(1, bad_symbol_heuristic_shift(stop_table, pattern_idx, text[text_idx].word));
    }

    return answer;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    Text text;
    Pattern pattern;

    std::string line;
    std::getline(std::cin, line);
    to_lower_case(line);

    std::stringstream ss{line};
    std::string word;
    while(ss >> word) {
        pattern.push_back(word);
    }

    uint line_idx = 1;
    while(std::getline(std::cin, line)) {
        if (line.empty()) {
            line_idx++;
            continue;
        }

        to_lower_case(line);

        ss = std::stringstream(line);
        uint word_idx = 1;
        while(ss >> word) {
            text.push_back({word, line_idx, word_idx});
            word_idx++;
        }

        line_idx++;
    }

    for(auto [_line_idx, _word_idx] : search(text, pattern)) {
        std::cout << _line_idx << ", " << _word_idx << "\n";
    }
}
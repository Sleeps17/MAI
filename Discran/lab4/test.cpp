#include "gtest/gtest.h"
#include <random>

constexpr uint min_count = 30;
constexpr uint max_count = 1000000;

const int MAX = 256;

const static std::vector<char> alphabet = {
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
};

const static std::vector<char> available_delims_for_pattern = {' '};

const static std::vector<char> available_delims_for_text = {' ', '\n'};

uint get_random_number() {
    std::random_device rd;
    std::mt19937 gen(rd());
    return static_cast<uint>(std::uniform_int_distribution<uint>(min_count, max_count)(gen));
}

std::string get_random_word(uint size) {
    std::string word;

    for (uint i = 0; i < size; ++i) {
        word += alphabet[get_random_number() % alphabet.size()];
    }

    return word;
}

std::string generate_pattern() {
    uint count_words = get_random_number() % 30;
    std::string pattern;

    for (uint i = 0; i < count_words; ++i) {
        pattern += get_random_word(16);

        if (i != count_words - 1) {
            pattern += available_delims_for_pattern[get_random_number() % available_delims_for_pattern.size()];
        }
    }

    return pattern;
}

std::string generate_text() {
    uint count_words = get_random_number();

    std::string text;
    for (uint i = 0; i < count_words; ++i) {
        text += get_random_word(16);

        if (i != count_words - 1) {
            text += available_delims_for_text[get_random_number() % available_delims_for_text.size()];
        }
    }

    return text;
}

void badCharHeuristic(const std::string& str, int size, int bad_char[MAX]) {
    for (int i = 0; i < MAX; i++) {
        bad_char[i] = size;
    }
    for (int i = 0; i < size - 1; i++) {
        bad_char[(int) str[i]] = size - i - 1;
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
        while (j >= 0 && tolower(pat[j]) == tolower(txt[s + j])) {
            j--;
        }
        if (j < 0) {
            answer.push_back(s);
            s += m;
        } else {
            s += bad_char[(int) tolower(txt[s + j])] - m + j + 1 >= 1 ? bad_char[(int) tolower(txt[s + j])] - m + j + 1 : 1;
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

TEST(Test_BM, Test1) {
    auto pattern = generate_pattern();
    auto text = generate_text();

    auto stupid_positions = findPattern(text, pattern);
    auto positions = search(text, pattern);

    EXPECT_EQ(stupid_positions.size(), positions.size());

    for (uint i = 0; i < positions.size(); ++i) {
        EXPECT_EQ(stupid_positions[i], positions[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
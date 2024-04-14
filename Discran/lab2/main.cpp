#include <iostream>
#include <fstream>
#include "patricia/patricia.hpp"

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
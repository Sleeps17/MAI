#include <iostream>
#include "patricia/patricia.hpp"

void to_lower_case(std::string& str) {
    for(int i = 0; i < str.length(); i++) {
        str[i] = (char)tolower(str[i]);
    }
}

int main() {
    std::string cmd;
    Patricia p{};

    while(std::cin >> cmd) {
        bool catched = false;
        if (cmd == "+") {
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
                std::cout << "Exists" << '\n';
            }

            if (!catched) {
                std::cout << "OK" << '\n';
            }
        } else if (cmd == "-") {
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
        } else {
            uint64_t value;
            to_lower_case(cmd);
            try {
                value = p.at(cmd);
            }
            catch(std::runtime_error& ex) {
                catched = true;
                std::cout << "NoSuchWord" << '\n';
            }

            if (!catched) {
                std::cout << "OK: " << value << '\n';
            }
        }
    }
}
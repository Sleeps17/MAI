#ifndef RESPONSE_HPP
#define RESPONSE_HPP

#include <string>
#include <chrono>

enum Status {
    OK,
    ERROR,
};


struct Response {
public:
    Status status;
    int result{};
    int pid{};
    std::string error{};
    std::vector<int> unavailable;

    Response(): status(OK) {}
    explicit Response(Status status): status(status) {}
    explicit Response(const std::string& status) {
        if (status == "OK") {
            this->status = OK;
        } else {
            this->status = ERROR;
        }
    }

    explicit operator std::string() const {
        if (status == OK) {
            return "OK";
        } else {
            return "ERROR";
        }
    }

    [[nodiscard]] bool StatusOK() const {
        return status == OK;
    }
};

#endif //RESPONSE_HPP

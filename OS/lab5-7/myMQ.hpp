#ifndef myMQ_hpp
#define myMQ_hpp

#include <zmq.hpp>

const int BASE_PORT = 5505;

std::string MakeConnPort(int id) {
    return "tcp://localhost:" + std::to_string(BASE_PORT + id);
}

std::string MakeBindPort(int id) {
    return "tcp://*:" + std::to_string(BASE_PORT + id);
}


#endif //myMQ_hpp
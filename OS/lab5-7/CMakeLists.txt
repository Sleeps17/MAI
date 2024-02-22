cmake_minimum_required(VERSION 3.20)
project(lab5-7)

set(CMAKE_CXX_STANDARD 20)
add_definitions(-lzmq)

add_executable(control controlNode.cpp)

add_executable(calculation calculationNode.cpp)

target_link_libraries(control zmq)
target_link_libraries(calculation zmq)

set_target_properties(control PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set_target_properties(calculation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

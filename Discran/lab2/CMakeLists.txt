cmake_minimum_required(VERSION 3.22)
project(lab2)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -Wall")

add_executable(lab2_exe main.cpp patricia/patricia.cpp)

set_target_properties(lab2_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

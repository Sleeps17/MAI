CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
project(kp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(headers)

add_executable(kp_exe main.cpp)

set_target_properties(kp_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
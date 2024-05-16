cmake_minimum_required(VERSION 3.22)
project(lab4)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

add_executable(lab4_exe main.cpp)
add_executable(lab4_test test.cpp)

target_include_directories(lab4_test PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_link_libraries(lab4_test PRIVATE gtest gtest_main)

add_test(NAME lab1_test COMMAND lab1_test)

set_target_properties(lab4_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set_target_properties(lab4_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

cmake_minimum_required(VERSION 3.22)
project(lab1)

enable_testing()

add_executable(lab1_exe main.cpp)
add_executable(lab1_test test.cpp)

target_include_directories(lab1_test PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_link_libraries(lab1_test PRIVATE gtest gtest_main)

add_test(NAME lab1_test COMMAND lab1_test)

set_target_properties(lab1_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set_target_properties(lab1_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

cmake_minimum_required(VERSION 3.22)
project(lab2)

enable_testing()

add_executable(lab2_exe main.cpp patricia/patricia.cpp)
add_executable(lab2_test test.cpp
        patricia/patricia.cpp)

target_include_directories(lab2_test PRIVATE ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
target_link_libraries(lab2_test PRIVATE gtest gtest_main)

add_test(NAME lab2_test COMMAND lab2_test)

set_target_properties(lab2_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set_target_properties(lab2_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
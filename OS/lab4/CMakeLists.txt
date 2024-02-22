cmake_minimum_required(VERSION 3.10)
project(lab4)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR})

add_library(function1 SHARED function1.cpp)
add_library(function2 SHARED function2.cpp)

add_executable(staticMain staticMain.cpp
        function.hpp)
add_executable(dynamicMain dynamicMain.cpp
        function.hpp)

target_link_libraries(staticMain function1)
target_link_libraries(dynamicMain function1 function2)

set_target_properties(staticMain PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

set_target_properties(dynamicMain PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
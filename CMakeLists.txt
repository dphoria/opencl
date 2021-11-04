cmake_minimum_required(VERSION 3.1)
# search <package>_ROOT in find_package()
cmake_policy(SET CMP0074 NEW)
project(LearnOpenCL C CXX)

set(OpenCL_ROOT /usr/lib/x86_64-linux-gnu/)
find_package(OpenCL REQUIRED)

set(CMAKE_CXX_CLANG_TIDY clang-tidy)

include_directories(${OpenCL_INCLUDE_DIRS})
add_subdirectory(platform)
add_subdirectory(test)

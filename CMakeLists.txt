cmake_minimum_required(VERSION 3.1)
# search <package>_ROOT in find_package()
cmake_policy(SET CMP0074 NEW)
project(LearnOpenCL C CXX)

set(OpenCL_ROOT /usr/lib/x86_64-linux-gnu/)
find_package(OpenCL 2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
# opencl will use highest available if not specified
add_compile_definitions(CL_TARGET_OPENCL_VERSION=200)

include("conf/clang-tidy.cmake")
include("conf/clang-format.cmake")

include_directories(${OpenCL_INCLUDE_DIRS})
add_subdirectory(core)
add_subdirectory(test)

make_clang_format_target(d-ocl-platform test-opencl)
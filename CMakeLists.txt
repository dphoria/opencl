cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0074 NEW)
project(OCLDriver C CXX)

set(OpenCL_ROOT /usr/lib/x86_64-linux-gnu/)
find_package(OpenCL REQUIRED)

cmake_minimum_required(VERSION 3.1)
# search <package>_ROOT in find_package()
cmake_policy(SET CMP0074 NEW)
project(LearnOpenCL C CXX)

set(OpenCL_ROOT /usr/lib/x86_64-linux-gnu/)
find_package(OpenCL REQUIRED)

find_file(CLANG_TIDY_PATH
    clang-tidy
)
if (EXISTS "${CLANG_TIDY_PATH}")
    set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,clang-analyzer-*,modernize-*,readability-*)

    set(CLANG_TIDY_CMD_FILE "compile_commands.json")
    # TODO: don't know why file(CREATE_LINK SYMBOLIC) doesn't work
    file(COPY
        "${CMAKE_BINARY_DIR}/${CLANG_TIDY_CMD_FILE}"
        DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}"
    )
else()
    message("linting disabled")
endif()

include_directories(${OpenCL_INCLUDE_DIRS})
add_subdirectory(platform)
add_subdirectory(test)

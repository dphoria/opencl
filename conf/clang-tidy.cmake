find_file(CLANG_TIDY_PATH
    clang-tidy
)

if (EXISTS "${CLANG_TIDY_PATH}")
    # use .clang-tidy
    set(CMAKE_CXX_CLANG_TIDY clang-tidy -p "${CMAKE_SOURCE_DIR}/build")
else()
    message("clang-tidy not found")
endif()

find_file(CLANG_TIDY_PATH
    clang-tidy
)

if (EXISTS "${CLANG_TIDY_PATH}")
    # build this to show clang-tidy suggestions
    add_custom_target(check-clang-tidy)
    add_custom_command(
        TARGET check-clang-tidy PRE_BUILD
        COMMAND python scripts/clang_tidy.py --recurse -p "${CMAKE_BINARY_DIR}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )

    # build this to call clang-tidy --fix
    add_custom_target(apply-clang-tidy)
    add_custom_command(
        TARGET apply-clang-tidy PRE_BUILD
        # write corrections to file
        COMMAND python scripts/clang_tidy.py --recurse --write -p "${CMAKE_BINARY_DIR}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
else()
    message("clang-tidy not found")
endif()

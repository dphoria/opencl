set(CLANG_FORMAT_SRC_FILES "clang-format-srcs")

macro(setup_clang_format target)
    # save source file paths in disk file that the script will read
    file(WRITE "${CLANG_FORMAT_SRC_FILES}" "")

    # absolute source file path per line in file
    foreach(file_name ${ARGN})
        file(APPEND "${CLANG_FORMAT_SRC_FILES}" "${CMAKE_CURRENT_LIST_DIR}/${file_name}\n")
    endforeach()
endmacro()

# build this to show clang-format diff
add_custom_target(check-clang-format)
add_custom_command(
    TARGET check-clang-format PRE_BUILD
    COMMAND python scripts/clang_format.py --recurse
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
# also show clang-tidy suggestions when showing clang-format diff
add_dependencies(check-clang-format check-clang-tidy)

# build this to apply clang-format
add_custom_target(apply-clang-format)
add_custom_command(
    TARGET apply-clang-format PRE_BUILD
    # apply diff in-place
    COMMAND python scripts/clang_format.py --recurse --write
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# {ARGN} are expected to be project build targets
macro(make_clang_format_target)
    # show clang-format diff + build
    add_dependencies(check-clang-format ${ARGN})
endmacro()

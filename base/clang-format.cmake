set(CLANG_FORMAT_SRC_FILES "clang-format-srcs")

macro(setup_clang_format target)
    # save source file paths in disk file that the script will read
    file(WRITE "${CLANG_FORMAT_SRC_FILES}" "")

    # make string list of all parameters after the target name
    # expected to be source file paths relative to current dir
    # add to list as absolute paths
    foreach(file_name ${ARGN})
        file(APPEND "${CLANG_FORMAT_SRC_FILES}" "${CMAKE_CURRENT_LIST_DIR}/${file_name}\n")
    endforeach()

    add_dependencies(${target} check-clang-format)
endmacro()

# build this to show clang-format diff
add_custom_target(check-clang-format)
add_custom_command(
    TARGET check-clang-format PRE_BUILD
    COMMAND python scripts/clang-format.py --recurse
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# build this to apply clang-format diff to source files in-place
add_custom_target(apply-clang-format)
add_custom_command(
    TARGET apply-clang-format PRE_BUILD
    COMMAND python scripts/clang-format.py --recurse --write
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

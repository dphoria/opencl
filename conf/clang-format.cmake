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
    COMMAND python scripts/clang-format.py --recurse
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# build this to apply clang-format
add_custom_target(apply-clang-format)
add_custom_command(
    TARGET apply-clang-format PRE_BUILD
    # apply diff in-place
    COMMAND python scripts/clang-format.py --recurse --write
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_custom_target(clang-format-applied)
add_dependencies(clang-format-applied apply-clang-format)

# {ARGN} are expected to be the build targets
macro(make_clang_format_target)
    add_custom_target(targets-built)
    # build the targets using clang-formatted source files
    add_dependencies(targets-built clang-format-applied ${ARGN})

    # build this to clang-format + build
    add_custom_target(build-clang-format)
    add_dependencies(build-clang-format targets-built)
endmacro()

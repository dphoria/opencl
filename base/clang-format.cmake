set(CLANG_FORMAT_SRC_FILES "clang-format-srcs")

macro(make_clang_format_srcs target)
    # save source file paths in disk file that the script will read
    file(WRITE "${CLANG_FORMAT_SRC_FILES}" "")

    # make string list of all parameters after the target name
    # expected to be source file paths relative to current dir
    # add to list as absolute paths
    foreach(file_name ${ARGN})
        file(APPEND "${CLANG_FORMAT_SRC_FILES}" "${CMAKE_CURRENT_LIST_DIR}/${file_name}\n")
    endforeach()
endmacro()

macro(post_build_clang_format target)
    find_file(CLANG_FORMAT_PATH
        clang-format
    )

    # make string list of all parameters after the target name
    # expected to be source file paths relative to current dir
    # add to list as absolute paths
    set(file_path_list)
    foreach(file_name ${ARGN})
        list(APPEND file_path_list "${CMAKE_CURRENT_LIST_DIR}/${file_name}")
    endforeach()
    # join list with ; delimiter
    list(JOIN file_path_list
        ";"
        src_files
    )

    if (EXISTS "${CLANG_FORMAT_PATH}")
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            # pass paths as a single quoted string
            COMMAND python ${CMAKE_SOURCE_DIR}/scripts/clang-format.py --src-files "'${src_files}'"
        )
    else()
        message("clang-format not found")
    endif()
endmacro()
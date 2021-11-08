macro(post_build_clang_format target src_files)
    find_file(CLANG_FORMAT_PATH
        clang-format
    )

    if (EXISTS "${CLANG_FORMAT_PATH}")
        add_custom_command(
            TARGET ${target}
            POST_BUILD
            COMMAND python ${CMAKE_SOURCE_DIR}/scripts/clang-format.py --src-files "${src_files}"
        )
    else()
        message("clang-format not found")
    endif()
endmacro()
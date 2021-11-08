macro(post_build_clang_format target src_files)
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND python ${CMAKE_SOURCE_DIR}/scripts/clang-format.py --src-files "${src_files}"
    )
endmacro()
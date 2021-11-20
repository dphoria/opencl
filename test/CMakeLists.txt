# kernel source files here
add_compile_definitions(PROGRAM_SRC_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/program-sources")
set(SOURCES
    main.cpp
    programs/3.4-vector-add.cpp
    programs/3.4-vector-add.h
)
add_executable(test-opencl ${SOURCES})

target_link_libraries(test-opencl
    d-ocl-core
)

# save source file paths for clang-format.py
setup_clang_format(test-opencl ${SOURCES})
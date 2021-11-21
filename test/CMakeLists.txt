set(SOURCES
    main.cpp
)
add_executable(test-opencl ${SOURCES})

target_link_libraries(test-opencl
    d-ocl-core
    d-ocl-examples
)

# save source file paths for clang-format.py
setup_clang_format(test-opencl ${SOURCES})
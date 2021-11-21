# kernel source files here
add_compile_definitions(PROGRAM_SRC_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/program-sources")
set(SOURCES
    d_ocl_examples.cpp
    d_ocl_examples_defines.h
    programs/3.4-vector-add.cpp
    programs/3.4-vector-add.h
    programs/programs_defines.h
)
add_library(d-ocl-examples SHARED ${SOURCES})

target_link_libraries(d-ocl-examples
    ${OpenCL_LIBRARIES}
    d-ocl-core
)
target_compile_definitions(d-ocl-examples
    PRIVATE EXPORT_D_OCL_EXAMPLES
)

# save source file paths for clang-format.py
setup_clang_format(d-ocl-examples ${SOURCES})
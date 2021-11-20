set(SOURCES
    d_ocl.cpp
    d_ocl.h
    d_ocl_defines.h
)
add_library(d-ocl-core SHARED ${SOURCES})

target_link_libraries(d-ocl-core
    ${OpenCL_LIBRARIES}
)
target_compile_definitions(d-ocl-core
    PRIVATE EXPORT_D_OCL_PLATFORM
)

# save source file paths for clang-format.py
setup_clang_format(d-ocl-core ${SOURCES})
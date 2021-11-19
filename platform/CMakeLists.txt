set(SOURCES
    d_ocl_platform.cpp
    d_ocl_platform.h
)
add_library(d-ocl-platform SHARED ${SOURCES})

target_link_libraries(d-ocl-platform
    ${OpenCL_LIBRARIES}
)
target_compile_definitions(d-ocl-platform
    PRIVATE EXPORT_D_OCL_PLATFORM
)

# save source file paths for clang-format.py
setup_clang_format(d-ocl-platform ${SOURCES})
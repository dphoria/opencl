set(SOURCES
    d_ocl_platform.cpp
    d_ocl_platform.h
)
add_library(DOclPlatform SHARED ${SOURCES})

target_link_libraries(DOclPlatform
    ${OpenCL_LIBRARIES}
)
target_compile_definitions(DOclPlatform
    PRIVATE EXPORT_D_OCL_PLATFORM
)

# save source file paths for clang-format.py
setup_clang_format(DOclPlatform ${SOURCES})
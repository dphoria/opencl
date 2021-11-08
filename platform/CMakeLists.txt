set(D_OCL_FILES
    d_ocl_platform.cpp
    d_ocl_platform.h
)
add_library(DOclPlatform SHARED
    ${D_OCL_FILES}
)

target_link_libraries(DOclPlatform
    ${OpenCL_LIBRARIES}
)
target_compile_definitions(DOclPlatform
    PRIVATE EXPORT_D_OCL_PLATFORM
)

post_build_clang_format(DOclPlatform ${D_OCL_FILES})
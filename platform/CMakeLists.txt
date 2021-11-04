add_library(DOclPlatform SHARED
    d_ocl_platform.cpp
    d_ocl_platform.h
)

target_link_libraries(DOclPlatform
    ${OpenCL_LIBRARIES}
)
target_compile_definitions(DOclPlatform
    PRIVATE EXPORT_D_OCL_PLATFORM
)
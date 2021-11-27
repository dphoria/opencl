set(SOURCES
    d_ocl.cpp
    d_ocl.h
    d_ocl_utils.cpp
    d_ocl_utils.h
    d_ocl_defines.h
)
add_library(d-ocl-core SHARED ${SOURCES})

include_directories("${OpenCV_INCLUDE_DIRS}")

target_link_libraries(d-ocl-core
    ${OpenCL_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_definitions(d-ocl-core
    PRIVATE EXPORT_D_OCL_CORE
)

# save source file paths for clang_format.py
setup_clang_format(d-ocl-core ${SOURCES})
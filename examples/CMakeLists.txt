# kernel source files here
add_compile_definitions(EX_RESOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/res")
set(SOURCES
    d_ocl_examples.cpp
    d_ocl_examples.h
)
# add all source files in src/
file(GLOB_RECURSE EX_SRC_FILES FOLLOW_SYMLINKS                  # visit subdirectories that are symlinks
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS    # regenerate build system when file list changes
    src/*.cpp
    src/*.h
)

list(APPEND SOURCES ${EX_SRC_FILES})
add_library(d-ocl-examples SHARED ${SOURCES})

include_directories("${OpenCV_INCLUDE_DIRS}")

target_link_libraries(d-ocl-examples
    ${OpenCL_LIBRARIES}
    ${OpenCV_LIBS}
    d-ocl-core
)
target_compile_definitions(d-ocl-examples
    PRIVATE EXPORT_D_OCL_EXAMPLES
)

# save source file paths for clang_format.py
setup_clang_format(d-ocl-examples ${SOURCES})
add_executable(test-opencl
    main.cpp
)

target_link_libraries(test-opencl
    DOclPlatform
)

post_build_clang_format(test-opencl main.cpp)
add_executable(test-opencl
    main.cpp
    #programs/3.4-vector-add.cpp
    #programs/3.4-vector-add.h
)

target_link_libraries(test-opencl
    DOclPlatform
)

post_build_clang_format(test-opencl main.cpp)
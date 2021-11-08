add_executable(test-opencl
    main.cpp
)

target_link_libraries(test-opencl
    DOclPlatform
)

post_build_clang_format(test-opencl ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
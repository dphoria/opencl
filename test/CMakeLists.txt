add_executable(test-opencl
    main.cpp
)

target_link_libraries(test-opencl
    DOclPlatform
)

#include "../platform/d_ocl_platform.h"
#include <iostream>

int main(int argc, char** argv)
{
    int retval = 0;

    cl_platform_id platform;
    if (findGpuPlatform(&platform)) {
        std::cout << "found gpu platform" << std::endl;
    } else {
        retval = 1;
        std::cerr << "did not find gpu platform" << std::endl;
    }

    return retval;
}

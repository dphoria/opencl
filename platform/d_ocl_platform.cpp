#include "d_ocl_platform.h"


bool findGpuPlatform(cl_platform_id* platform)
{
    return (clGetPlatformIDs(1, platform, nullptr) == CL_SUCCESS);
}

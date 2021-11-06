#include "d_ocl_platform.h"


auto findGpuPlatform(cl_platform_id* platform) -> bool
{
    return (clGetPlatformIDs(1, platform, nullptr) == CL_SUCCESS);
}

auto findGpuDevice(cl_platform_id platform, cl_device_id* device) -> bool
{
    return (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, nullptr) == CL_SUCCESS);
}

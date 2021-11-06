#ifndef D_OCL_PLATFORM_H
#define D_OCL_PLATFORM_H

#include <CL/cl.h>
#include "d_ocl_platform_defines.h"


// return the first gpu platform found
auto D_OCL_PLATFORM_API findGpuPlatform(cl_platform_id* platform) -> bool;
// return the first gpu device in platform
auto D_OCL_PLATFORM_API findGpuDevice(cl_platform_id platform, cl_device_id* device) -> bool;

#endif  // D_OCL_PLATFORM_H

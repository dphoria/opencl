#ifndef D_OCL_PLATFORM_H
#define D_OCL_PLATFORM_H

#include <CL/cl.h>
#include "d_ocl_platform_defines.h"


// return the first gpu platform found
bool D_OCL_PLATFORM_API findGpuPlatform(cl_platform_id* platform);

#endif  // D_OCL_PLATFORM_H

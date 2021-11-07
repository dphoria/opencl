#ifndef D_OCL_PLATFORM_H
#define D_OCL_PLATFORM_H


#include <vector>
#include <unordered_map>
#include <string>
#include <CL/cl.h>
#include "d_ocl_platform_defines.h"


auto D_OCL_PLATFORM_API gpuPlatforms() -> std::vector<cl_platform_id>;
auto D_OCL_PLATFORM_API gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>;
// every vector<cl_device_id> is guaranteed to have at least 1 cl_device_id
auto D_OCL_PLATFORM_API gpuPlatformDevices() -> std::unordered_map<cl_platform_id, std::vector<cl_device_id>>;

auto D_OCL_PLATFORM_API description(cl_device_id device) -> std::string;


#endif  // D_OCL_PLATFORM_H

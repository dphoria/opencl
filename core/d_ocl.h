#ifndef D_OCL_PLATFORM_H
#define D_OCL_PLATFORM_H

#include "d_ocl_defines.h"
#include <CL/cl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

auto D_OCL_API gpuPlatforms() -> std::vector<cl_platform_id>;
auto D_OCL_API gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>;
// every vector<cl_device_id> is guaranteed to have at least 1 cl_device_id
auto D_OCL_API gpuPlatformDevices()
    -> std::unordered_map<cl_platform_id, std::vector<cl_device_id>>;

// d_ocl_context::context is auto released when finished
auto D_OCL_API createContext(cl_platform_id platform,
                             const std::vector<cl_device_id>& devices)
    -> std::shared_ptr<d_ocl_context>;
auto D_OCL_API createCmdQueue(cl_device_id device, cl_context context)
    -> cl_command_queue;

// read kernel source from filePath to create cl_program
auto D_OCL_API createProgram(cl_context context, const std::string& filePath)
    -> cl_program;

auto D_OCL_API description(cl_device_id device) -> std::string;

#endif // D_OCL_PLATFORM_H

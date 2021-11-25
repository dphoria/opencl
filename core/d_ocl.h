#ifndef D_OCL_H
#define D_OCL_H

#include "d_ocl_defines.h"
#include <CL/cl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace d_ocl {
// helper to return true if funcRetval == CL_SUCCESS
// else print funcRetval and return false
auto D_OCL_API check_run(const std::string& funcName, cl_int funcRetval)
    -> bool;

auto D_OCL_API gpuPlatforms() -> std::vector<cl_platform_id>;
auto D_OCL_API gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>;
// every vector<cl_device_id> is guaranteed to have at least 1 cl_device_id
auto D_OCL_API gpuPlatformDevices()
    -> std::unordered_map<cl_platform_id, std::vector<cl_device_id>>;

// the following "created" resources like cl_context
// are managed and auto released via shared_ptr

auto D_OCL_API createContext(cl_platform_id platform,
                             const std::vector<cl_device_id>& devices)
    -> std::shared_ptr<d_ocl_manager<cl_context>>;
auto D_OCL_API createCmdQueue(cl_device_id device, cl_context context)
    -> std::shared_ptr<d_ocl_manager<cl_command_queue>>;

struct D_OCL_API basic_palette
{
    std::shared_ptr<d_ocl_manager<cl_context>> context;
    std::shared_ptr<d_ocl_manager<cl_command_queue>> cmdQueue;
};
// convenience func to create context and command queue for the first gpu device
// found
auto D_OCL_API createBasicPalette(basic_palette& palette) -> bool;

// read kernel source from filePath to create cl_program
// program will have been built (compile, link)
auto D_OCL_API createProgram(cl_context context, const std::string& filePath)
    -> std::shared_ptr<d_ocl_manager<cl_program>>;

// generate human-readable description
auto D_OCL_API description(cl_device_id device) -> std::string;
} // namespace d_ocl

#endif // D_OCL_H

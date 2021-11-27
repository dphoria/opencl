#ifndef D_OCL_H
#define D_OCL_H

#include "d_ocl_defines.h"
#include <CL/cl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cv {
struct Mat;
}

namespace d_ocl {

// ensure release when finished with open resource like cl_context
template<typename T>
struct manager
{
    // e.g. clReleaseContext
    using opencl_release_func = cl_int (*)(T);

    static auto makeShared(T openclObject, opencl_release_func releaseFunc)
        -> std::shared_ptr<manager<T>>
    {
        if (openclObject == nullptr) {
            // operator bool will fail test
            return std::shared_ptr<manager<T>>();
        }

        return std::make_shared<manager<T>>(openclObject, releaseFunc);
    }

    manager(T openclObject, opencl_release_func releaseFunc)
    {
        this->openclObject = openclObject;
        this->releaseFunc = releaseFunc;
    }
    ~manager()
    {
        if (openclObject != nullptr && releaseFunc != nullptr) {
            releaseFunc(openclObject);
        }
    }

    T openclObject{nullptr};
    opencl_release_func releaseFunc{nullptr};
};

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
    -> std::shared_ptr<manager<cl_context>>;
auto D_OCL_API createCmdQueue(cl_device_id device, cl_context context)
    -> std::shared_ptr<manager<cl_command_queue>>;

struct D_OCL_API basic_palette
{
    std::shared_ptr<manager<cl_context>> context;
    std::shared_ptr<manager<cl_command_queue>> cmdQueue;
};
// convenience func to create context and command queue for the first gpu device
// found
auto D_OCL_API createBasicPalette(basic_palette& palette) -> bool;

// read kernel source from filePath to create cl_program
// program will have been built (compile, link)
auto D_OCL_API createProgram(cl_context context, const std::string& filePath)
    -> std::shared_ptr<manager<cl_program>>;

// read image at filePath and initialize device-side image object with the input
// image. opencvMat will be set to the loaded image if not null.
auto D_OCL_API createInputImage(cl_context context,
                                cl_mem_flags flags,
                                const std::string& filePath,
                                cv::Mat* opencvMat = nullptr)
    -> std::shared_ptr<manager<cl_mem>>;

// wrapper around clGetDeviceInfo()
// will query value count for param_name first then call param_value.resize()
template<typename T>
auto D_OCL_API information(cl_device_id device,
                           cl_device_info param_name,
                           std::vector<T>& param_value,
                           T default_value) -> bool;
// generate human-readable description
auto D_OCL_API description(cl_device_id device) -> std::string;
} // namespace d_ocl

#endif // D_OCL_H

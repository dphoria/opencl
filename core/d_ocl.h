#ifndef D_OCL_H
#define D_OCL_H

#include "d_ocl_defines.h"
#include "d_ocl_utils.h"
#include <CL/cl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cv {
struct Mat;
}

namespace d_ocl {
auto D_OCL_API gpuPlatforms() -> std::vector<cl_platform_id>;
auto D_OCL_API gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>;
// every vector<cl_device_id> is guaranteed to have at least 1 cl_device_id
auto D_OCL_API gpuPlatformDevices()
    -> std::unordered_map<cl_platform_id, std::vector<cl_device_id>>;

// the following "created" resources like cl_context
// are managed and auto released via shared_ptr

auto D_OCL_API createContext(cl_platform_id platform,
                             const std::vector<cl_device_id>& devices)
    -> std::shared_ptr<utils::manager<cl_context>>;
auto D_OCL_API createCmdQueue(cl_device_id device, cl_context context)
    -> std::shared_ptr<utils::manager<cl_command_queue>>;

struct D_OCL_API context_set
{
    std::shared_ptr<utils::manager<cl_context>> context;
    std::shared_ptr<utils::manager<cl_command_queue>> cmdQueue;
};
// convenience func to create context and command queue for the first gpu device
auto D_OCL_API createContextSet(context_set& contextSet) -> bool;

// read kernel source from filePath to create cl_program
// program will have been built (compile, link)
auto D_OCL_API createProgram(cl_context context, const std::string& filePath)
    -> std::shared_ptr<utils::manager<cl_program>>;

// read image at filePath and initialize device-side image object with the input
// image. opencvMat will be set to the loaded image if not null.
auto D_OCL_API createInputImage(
    cl_context context,
    cl_mem_flags flags,
    const std::string& filePath,
    // pass converters like toRgba to apply on the input cv::Mat
    // before setting up cl_mem
    const std::vector<utils::mat_convert_func>& matConverts,
    cv::Mat* opencvMat = nullptr) -> std::shared_ptr<utils::manager<cl_mem>>;
// create device-side output buffer for image with same specification
// (resolution, etc.) as opencvMat
auto D_OCL_API createOutputImage(cl_context context,
                                 cl_mem_flags flags,
                                 const cv::Mat& opencvMat)
    -> std::shared_ptr<utils::manager<cl_mem>>;
} // namespace d_ocl

#endif // D_OCL_H

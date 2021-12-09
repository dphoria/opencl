#ifndef D_OCL_UTILS_H
#define D_OCL_UTILS_H

#include "d_ocl_defines.h"
#include <CL/cl.h>
#include <functional>
#include <vector>

namespace cv {
struct Mat;
}

namespace d_ocl {
namespace utils {
// return string representation for code
// e.g. CL_DEVICE_NOT_FOUND -> "CL_DEVICE_NOT_FOUND"
auto D_OCL_API errorString(cl_int code) -> std::string;
// helper to return true if funcRetval == CL_SUCCESS
// else print funcRetval and return false
auto D_OCL_API checkRun(const std::string& funcName, cl_int funcRetval) -> bool;

// ensure release when finished with open resource like cl_context
template<typename T>
struct manager
{
    // e.g. clReleaseContext
    using opencl_release_func = std::function<cl_int(T)>;

    static auto makeShared(T openclObject, opencl_release_func releaseFunc)
        -> std::shared_ptr<manager<T>>;

    manager(T openclObject, opencl_release_func releaseFunc);
    ~manager();

    T openclObject{nullptr};
    opencl_release_func releaseFunc{nullptr};
};

using mat_convert_func = std::function<bool(const cv::Mat*, cv::Mat*)>;
// change channel order from opencv-default bgra to rgba
// rgbaMat will always be 1 or 4 channels.
// rgb has more restrictions of compatible data type than rgba in opencl
auto D_OCL_API toRgba(const cv::Mat* bgraMat, cv::Mat* rgbaMat) -> bool;
// convert data depth to 32-bit float 0.0~1.0
auto D_OCL_API toFloat(const cv::Mat* inMat, cv::Mat* floatMat) -> bool;

// wrapper around clGetDeviceInfo()
// will query value count for param_name first then call param_value.resize()
template<typename T>
auto D_OCL_API information(cl_device_id device,
                           cl_device_info param_name,
                           std::vector<T>& param_value,
                           T default_value) -> bool;
// generate human-readable description
auto D_OCL_API description(cl_device_id device) -> std::string;

// max compute units = max work groups
auto D_OCL_API maxComputeUnits(cl_device_id device) -> cl_uint;
// convenience func for maximum possible # work-items in a work-group per
// dimension
auto D_OCL_API maxWorkGroupSize(cl_device_id device) -> std::vector<size_t>;
} // namespace utils
} // namespace d_ocl

#include "d_ocl_manager.cpp"
#endif
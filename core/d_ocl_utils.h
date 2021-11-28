#ifndef D_OCL_UTILS_H
#define D_OCL_UTILS_H

#include "d_ocl_defines.h"
#include <CL/cl.h>
#include <vector>

namespace cv {
struct Mat;
}

namespace d_ocl {
namespace utils {
// helper to return true if funcRetval == CL_SUCCESS
// else print funcRetval and return false
auto D_OCL_API check_run(const std::string& funcName, cl_int funcRetval)
    -> bool;

// ensure release when finished with open resource like cl_context
template<typename T>
struct manager
{
    // e.g. clReleaseContext
    using opencl_release_func = int (*)(T);

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
            (*releaseFunc)(openclObject);
        }
    }

    T openclObject{nullptr};
    opencl_release_func releaseFunc{nullptr};
};

using mat_convert_func = bool (*)(const cv::Mat*, cv::Mat*);
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
} // namespace utils
} // namespace d_ocl

#endif
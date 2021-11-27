#include "d_ocl_utils.h"
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

auto d_ocl::utils::check_run(const std::string& funcName, cl_int funcRetval)
    -> bool
{
    if (funcRetval == CL_SUCCESS) {
        return true;
    }

    std::cerr << funcName << "() -> " << funcRetval << std::endl;
    return false;
}

auto d_ocl::utils::toRgba(const cv::Mat* bgraMat, cv::Mat* rgbaMat) -> bool
{
    *rgbaMat = *bgraMat;
    int conversionCode;

    // change channel order from opencv default bgra to common rgba
    if (rgbaMat->channels() == 3) {
        conversionCode = cv::COLOR_BGR2RGB;
    } else if (rgbaMat->channels() == 4) {
        conversionCode = cv::COLOR_BGRA2RGBA;
    } else {
        // no channel reordering if not bgr[a]
        return true;
    }

    cv::cvtColor(*bgraMat, *rgbaMat, conversionCode);
    return true;
}

auto d_ocl::utils::toFloat(const cv::Mat* intMat, cv::Mat* floatMat) -> bool
{
    double scale = 1;
    double offset = 0;

    // input * scale + offset -> output

    switch (intMat->depth()) {
    case CV_8S:
        // -128 -> -0.5
        scale = 1.0 / (1 << 8);
        // -0.5 -> 0
        offset = 0.5;
        break;
    case CV_8U:
        // 255 -> 1.0
        scale = 1.0 / std::numeric_limits<uint8_t>::max();
        break;
    case CV_16S:
        scale = 1.0 / (1 << 16);
        offset = 0.5;
        break;
    case CV_16U:
        scale = 1.0 / std::numeric_limits<uint16_t>::max();
        break;
    case CV_32S:
        scale = 1.0 / ((int64_t)1 << 32);
        offset = 0.5;
        break;
    case CV_16F:
    case CV_64F:
        // just * 1.0
        break;
    case CV_32F:
        // nothing to do
        *floatMat = *intMat;
        return true;
    default:
        std::cerr << "unrecognized cv::Mat::depth() -> " << intMat->depth()
                  << " for input image to " << __FUNCTION__ << std::endl;
    }

    intMat->convertTo(
        *floatMat, CV_MAKETYPE(CV_32F, intMat->channels()), scale, offset);
    return true;
}

// wrapper around clGetDeviceInfo()
template<typename T>
auto d_ocl::utils::information(cl_device_id device,
                               cl_device_info param_name,
                               std::vector<T>& param_value,
                               T default_value) -> bool
{
    // first find out value string length
    size_t requiredSize = 0;
    // requiredSize will be set to value string length
    if (!check_run(
            "clGetDeviceInfo",
            clGetDeviceInfo(device, param_name, 0, nullptr, &requiredSize))) {
        return false;
    }

    // add 1 for safety, like null-termination
    param_value.resize(requiredSize + 1, default_value);
    return check_run(
        "clGetDeviceInfo",
        clGetDeviceInfo(
            device, param_name, requiredSize, param_value.data(), nullptr));
};

auto d_ocl::utils::description(cl_device_id device) -> std::string
{
    std::ostringstream stream;

    // print like
    // parameter name: value
    // parameter name: value1 value2

    std::vector<cl_device_info> paramIds
        = {CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DEVICE_VERSION};
    std::vector<std::string> paramNames
        = {"CL_DEVICE_NAME", "CL_DEVICE_VENDOR", "CL_DEVICE_VERSION"};
    for (int i = 0; i < paramIds.size(); i++) {
        stream << paramNames[i] << ": ";

        std::vector<char> value;
        if (information<char>(device, paramIds[i], value, '\0')) {
            stream << value.data();
        }
        stream << std::endl;
    }

    paramIds = {CL_DEVICE_MAX_COMPUTE_UNITS,
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                CL_DEVICE_MAX_WORK_ITEM_SIZES};
    paramNames = {"CL_DEVICE_MAX_COMPUTE_UNITS",
                  "CL_DEVICE_MAX_WORK_GROUP_SIZE",
                  "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS",
                  "CL_DEVICE_MAX_WORK_ITEM_SIZES"};
    for (int i = 0; i < paramIds.size(); i++) {
        stream << paramNames[i] << ": ";

        std::vector<cl_uint> value;
        if (information<cl_uint>(device, paramIds[i], value, 0)) {
            for (const cl_uint& v : value) {
                stream << v << " ";
            }
        }
        stream << std::endl;
    }

    return stream.str();
}

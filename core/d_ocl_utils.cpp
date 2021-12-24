#include "d_ocl_utils.h"
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

auto d_ocl::utils::errorString(cl_int code) -> std::string
{
    switch (code) {
    case CL_DEVICE_NOT_FOUND:
        return __CL_STRINGIFY(CL_DEVICE_NOT_FOUND);
    case CL_DEVICE_NOT_AVAILABLE:
        return __CL_STRINGIFY(CL_DEVICE_NOT_AVAILABLE);
    case CL_COMPILER_NOT_AVAILABLE:
        return __CL_STRINGIFY(CL_COMPILER_NOT_AVAILABLE);
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return __CL_STRINGIFY(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    case CL_OUT_OF_RESOURCES:
        return __CL_STRINGIFY(CL_OUT_OF_RESOURCES);
    case CL_OUT_OF_HOST_MEMORY:
        return __CL_STRINGIFY(CL_OUT_OF_HOST_MEMORY);
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return __CL_STRINGIFY(CL_PROFILING_INFO_NOT_AVAILABLE);
    case CL_MEM_COPY_OVERLAP:
        return __CL_STRINGIFY(CL_MEM_COPY_OVERLAP);
    case CL_IMAGE_FORMAT_MISMATCH:
        return __CL_STRINGIFY(CL_IMAGE_FORMAT_MISMATCH);
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return __CL_STRINGIFY(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    case CL_BUILD_PROGRAM_FAILURE:
        return __CL_STRINGIFY(CL_BUILD_PROGRAM_FAILURE);
    case CL_MAP_FAILURE:
        return __CL_STRINGIFY(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        return __CL_STRINGIFY(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        return __CL_STRINGIFY(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
    case CL_COMPILE_PROGRAM_FAILURE:
        return __CL_STRINGIFY(CL_COMPILE_PROGRAM_FAILURE);
    case CL_LINKER_NOT_AVAILABLE:
        return __CL_STRINGIFY(CL_LINKER_NOT_AVAILABLE);
    case CL_LINK_PROGRAM_FAILURE:
        return __CL_STRINGIFY(CL_LINK_PROGRAM_FAILURE);
    case CL_DEVICE_PARTITION_FAILED:
        return __CL_STRINGIFY(CL_DEVICE_PARTITION_FAILED);
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        return __CL_STRINGIFY(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
    case CL_INVALID_VALUE:
        return __CL_STRINGIFY(CL_INVALID_VALUE);
    case CL_INVALID_DEVICE_TYPE:
        return __CL_STRINGIFY(CL_INVALID_DEVICE_TYPE);
    case CL_INVALID_PLATFORM:
        return __CL_STRINGIFY(CL_INVALID_PLATFORM);
    case CL_INVALID_DEVICE:
        return __CL_STRINGIFY(CL_INVALID_DEVICE);
    case CL_INVALID_CONTEXT:
        return __CL_STRINGIFY(CL_INVALID_CONTEXT);
    case CL_INVALID_QUEUE_PROPERTIES:
        return __CL_STRINGIFY(CL_INVALID_QUEUE_PROPERTIES);
    case CL_INVALID_COMMAND_QUEUE:
        return __CL_STRINGIFY(CL_INVALID_COMMAND_QUEUE);
    case CL_INVALID_HOST_PTR:
        return __CL_STRINGIFY(CL_INVALID_HOST_PTR);
    case CL_INVALID_MEM_OBJECT:
        return __CL_STRINGIFY(CL_INVALID_MEM_OBJECT);
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return __CL_STRINGIFY(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    case CL_INVALID_IMAGE_SIZE:
        return __CL_STRINGIFY(CL_INVALID_IMAGE_SIZE);
    case CL_INVALID_SAMPLER:
        return __CL_STRINGIFY(CL_INVALID_SAMPLER);
    case CL_INVALID_BINARY:
        return __CL_STRINGIFY(CL_INVALID_BINARY);
    case CL_INVALID_BUILD_OPTIONS:
        return __CL_STRINGIFY(CL_INVALID_BUILD_OPTIONS);
    case CL_INVALID_PROGRAM:
        return __CL_STRINGIFY(CL_INVALID_PROGRAM);
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return __CL_STRINGIFY(CL_INVALID_PROGRAM_EXECUTABLE);
    case CL_INVALID_KERNEL_NAME:
        return __CL_STRINGIFY(CL_INVALID_KERNEL_NAME);
    case CL_INVALID_KERNEL_DEFINITION:
        return __CL_STRINGIFY(CL_INVALID_KERNEL_DEFINITION);
    case CL_INVALID_KERNEL:
        return __CL_STRINGIFY(CL_INVALID_KERNEL);
    case CL_INVALID_ARG_INDEX:
        return __CL_STRINGIFY(CL_INVALID_ARG_INDEX);
    case CL_INVALID_ARG_VALUE:
        return __CL_STRINGIFY(CL_INVALID_ARG_VALUE);
    case CL_INVALID_ARG_SIZE:
        return __CL_STRINGIFY(CL_INVALID_ARG_SIZE);
    case CL_INVALID_KERNEL_ARGS:
        return __CL_STRINGIFY(CL_INVALID_KERNEL_ARGS);
    case CL_INVALID_WORK_DIMENSION:
        return __CL_STRINGIFY(CL_INVALID_WORK_DIMENSION);
    case CL_INVALID_WORK_GROUP_SIZE:
        return __CL_STRINGIFY(CL_INVALID_WORK_GROUP_SIZE);
    case CL_INVALID_WORK_ITEM_SIZE:
        return __CL_STRINGIFY(CL_INVALID_WORK_ITEM_SIZE);
    case CL_INVALID_GLOBAL_OFFSET:
        return __CL_STRINGIFY(CL_INVALID_GLOBAL_OFFSET);
    case CL_INVALID_EVENT_WAIT_LIST:
        return __CL_STRINGIFY(CL_INVALID_EVENT_WAIT_LIST);
    case CL_INVALID_EVENT:
        return __CL_STRINGIFY(CL_INVALID_EVENT);
    case CL_INVALID_OPERATION:
        return __CL_STRINGIFY(CL_INVALID_OPERATION);
    case CL_INVALID_GL_OBJECT:
        return __CL_STRINGIFY(CL_INVALID_GL_OBJECT);
    case CL_INVALID_BUFFER_SIZE:
        return __CL_STRINGIFY(CL_INVALID_BUFFER_SIZE);
    case CL_INVALID_MIP_LEVEL:
        return __CL_STRINGIFY(CL_INVALID_MIP_LEVEL);
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return __CL_STRINGIFY(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
    case CL_INVALID_PROPERTY:
        return __CL_STRINGIFY(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
    case CL_INVALID_IMAGE_DESCRIPTOR:
        return __CL_STRINGIFY(CL_INVALID_IMAGE_DESCRIPTOR);
    case CL_INVALID_COMPILER_OPTIONS:
        return __CL_STRINGIFY(CL_INVALID_COMPILER_OPTIONS);
    case CL_INVALID_LINKER_OPTIONS:
        return __CL_STRINGIFY(CL_INVALID_LINKER_OPTIONS);
    case CL_INVALID_DEVICE_PARTITION_COUNT:
        return __CL_STRINGIFY(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
    case CL_INVALID_PIPE_SIZE:
        return __CL_STRINGIFY(CL_INVALID_PIPE_SIZE);
    case CL_INVALID_DEVICE_QUEUE:
        return __CL_STRINGIFY(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
    case CL_INVALID_SPEC_ID:
        return __CL_STRINGIFY(CL_INVALID_SPEC_ID);
    case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
        return __CL_STRINGIFY(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif

    default:
        // don't know; just return the value
        return std::to_string(code);
    }
}

auto d_ocl::utils::checkRun(const std::string& funcName, cl_int funcRetval)
    -> bool
{
    if (funcRetval == CL_SUCCESS) {
        return true;
    }

    // function() -> CL_SOME_ERROR(n)
    std::cerr << funcName << "() -> " << errorString(funcRetval) << "("
              << funcRetval << ")" << std::endl;
    return false;
}

auto d_ocl::utils::toRgba(const cv::Mat* bgraMat, cv::Mat* rgbaMat) -> bool
{
    *rgbaMat = *bgraMat;
    int conversionCode;

    // change channel order from opencv default bgra to common rgba
    if (rgbaMat->channels() == 3) {
        conversionCode = cv::COLOR_BGR2RGBA;
    } else if (rgbaMat->channels() == 4) {
        conversionCode = cv::COLOR_BGRA2RGBA;
    } else {
        // no channel reordering if not bgr[a]
        return true;
    }

    cv::cvtColor(*bgraMat, *rgbaMat, conversionCode);
    return true;
}

auto d_ocl::utils::toGreyscale(const cv::Mat* inMat, cv::Mat* greyMat) -> bool
{
    *greyMat = *inMat;
    int conversionCode;

    if (inMat->channels() == 3) {
        conversionCode = cv::COLOR_BGR2GRAY;
    } else if (inMat->channels() == 4) {
        conversionCode = cv::COLOR_BGRA2GRAY;
    } else {
        return true;
    }

    cv::cvtColor(*inMat, *greyMat, conversionCode);
    return true;
}

auto d_ocl::utils::toFloat(const cv::Mat* inMat, cv::Mat* floatMat) -> bool
{
    double scale = 1;
    double offset = 0;

    // input * scale + offset -> output

    switch (inMat->depth()) {
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
        *floatMat = *inMat;
        return true;
    default:
        std::cerr << "unrecognized cv::Mat::depth() -> " << inMat->depth()
                  << " for input image to " << __FUNCTION__ << std::endl;
    }

    inMat->convertTo(
        *floatMat, CV_MAKETYPE(CV_32F, inMat->channels()), scale, offset);
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
    if (!checkRun(
            "clGetDeviceInfo",
            clGetDeviceInfo(device, param_name, 0, nullptr, &requiredSize))) {
        return false;
    }

    // add 1 for safety, like null-termination
    param_value.resize(requiredSize + 1, default_value);
    return checkRun(
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

auto d_ocl::utils::maxComputeUnits(cl_device_id device) -> cl_uint
{
    // # parallel compute units
    // keep in mind a work-group executes on a single compute unit
    std::vector<cl_uint> numComputeUnits;
    if (!information<cl_uint>(
            device, CL_DEVICE_MAX_COMPUTE_UNITS, numComputeUnits, 0)) {
        std::cerr << "error quering CL_DEVICE_MAX_COMPUTE_UNITS" << std::endl;
        return 0;
    }

    return numComputeUnits[0];
}

auto d_ocl::utils::maxWorkGroupSize(cl_device_id device) -> std::vector<size_t>
{
    // max # work-items per compute unit
    std::vector<size_t> maxWorkGroupSize;
    // max # work-items per dimension
    std::vector<size_t> maxWorkItemsByDim;
    // max dimensions
    std::vector<cl_uint> maxDimensions;

    if (!information<size_t>(
            device, CL_DEVICE_MAX_WORK_GROUP_SIZE, maxWorkGroupSize, 0)
        || !information<size_t>(
            device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemsByDim, 0)
        || !information<cl_uint>(
            device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, maxDimensions, 0)) {
        std::cerr << "error querying CL_DEVICE_MAX_WORK_GROUP_SIZE, "
                     "CL_DEVICE_MAX_WORK_ITEM_SIZES, "
                     "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS"
                  << std::endl;
        return std::vector<size_t>();
    }

    for (size_t& maxItems : maxWorkItemsByDim) {
        if (maxItems > maxWorkGroupSize[0]) {
            maxItems = maxWorkGroupSize[0];
        }
    }
    if (maxWorkItemsByDim.size() > maxDimensions[0]) {
        maxWorkItemsByDim.resize(maxDimensions[0]);
    }

    return maxWorkItemsByDim;
}

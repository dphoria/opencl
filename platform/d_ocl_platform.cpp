#include "d_ocl_platform.h"
#include <list>
#include <sstream>

auto gpuPlatforms() -> std::vector<cl_platform_id>
{
    // find available platform count first
    cl_uint numPlatforms;
    if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS) {
        numPlatforms = 0;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) !=
        CL_SUCCESS) {
        platforms.clear();
    }

    return platforms;
}

auto gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>
{
    cl_uint numDevices;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) !=
        CL_SUCCESS) {
        numDevices = 0;
    }

    std::vector<cl_device_id> devices(numDevices);
    if (clGetDeviceIDs(platform,
                       CL_DEVICE_TYPE_GPU,
                       numDevices,
                       devices.data(),
                       nullptr) != CL_SUCCESS) {
        devices.clear();
    }

    return devices;
}

auto gpuPlatformDevices()
    -> std::unordered_map<cl_platform_id, std::vector<cl_device_id>>
{
    // find platforms and all gpu devices in each

    std::unordered_map<cl_platform_id, std::vector<cl_device_id>>
        platformDevices;
    for (cl_platform_id platform : gpuPlatforms()) {
        std::vector<cl_device_id> devices = gpuDevices(platform);
        // exclude platform if no gpu device
        if (!devices.empty()) {
            platformDevices[platform] = devices;
        }
    }

    return platformDevices;
}

template <typename T>
auto information(cl_device_id device,
                 cl_device_info param_name,
                 std::vector<T> &param_value,
                 T default_value) -> bool
{
    // first find out value string length
    size_t requiredSize = 0;
    // requiredSize will be set to value string length
    if (clGetDeviceInfo(
            device, param_name, 0, param_value.data(), &requiredSize) !=
            CL_SUCCESS ||
        !requiredSize) {
        return false;
    }

    // add 1 for safety, like null-termination
    param_value.resize(requiredSize + 1, default_value);
    return (
        clGetDeviceInfo(
            device, param_name, requiredSize, param_value.data(), nullptr) ==
        CL_SUCCESS);
};

auto description(cl_device_id device) -> std::string
{
    std::ostringstream stream;

    std::vector<cl_device_info> paramIds = {
        CL_DEVICE_NAME, CL_DEVICE_VENDOR, CL_DEVICE_VERSION};
    std::vector<std::string> paramNames = {
        "CL_DEVICE_NAME", "CL_DEVICE_VENDOR", "CL_DEVICE_VERSION"};
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
            for (int j = 0; j < value.size(); j++) {
                stream << value[j];
                if (j) {
                    // space between numbers if multi-item value
                    stream << " ";
                }
            }
        }
        stream << std::endl;
    }

    return stream.str();
}

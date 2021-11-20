#include "d_ocl.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <sstream>

static std::mutex g_mutex;
// all-purpose buffer e.g. for file i/o
static std::vector<char> g_scratchBuffer;

auto gpuPlatforms() -> std::vector<cl_platform_id>
{
    // find available platform count first
    cl_uint numPlatforms;
    if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS) {
        numPlatforms = 0;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), nullptr)
        != CL_SUCCESS) {
        platforms.clear();
    }

    return platforms;
}

auto gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>
{
    cl_uint numDevices;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices)
        != CL_SUCCESS) {
        numDevices = 0;
    }

    std::vector<cl_device_id> devices(numDevices);
    if (clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr)
        != CL_SUCCESS) {
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

auto handleError(const char* errinfo,
                 const void* private_info,
                 size_t cb,
                 void* user_data) -> void
{
    // passed mutex* into clCreateContext in createContext
    std::lock_guard<std::mutex> lock(*reinterpret_cast<std::mutex*>(user_data));
    std::cerr << "opencl error:" << std::endl << errinfo << std::endl;
}

auto createContext(cl_platform_id platform,
                   const std::vector<cl_device_id>& devices)
    -> std::shared_ptr<d_ocl_manager<cl_context>>
{
    // list is terminated with 0. request to use this platform for the context
    std::vector<cl_context_properties> contextProperties
        = {CL_CONTEXT_PLATFORM,
           reinterpret_cast<cl_context_properties>(platform),
           0};

    return d_ocl_manager<cl_context>::makeShared(
        clCreateContext(contextProperties.data(),
                        devices.size(),
                        devices.data(),
                        // register callback to get errors
                        // during context creation
                        // and also at runtime for this context
                        &handleError,
                        &g_mutex,
                        nullptr),
        &clReleaseContext);
}

auto createCmdQueue(cl_device_id device, cl_context context)
    -> std::shared_ptr<d_ocl_manager<cl_command_queue>>
{
    return d_ocl_manager<cl_command_queue>::makeShared(
        clCreateCommandQueueWithProperties(context, device, nullptr, nullptr),
        &clReleaseCommandQueue);
}

auto createProgram(cl_context context, const std::string& filePath)
    -> std::shared_ptr<d_ocl_manager<cl_program>>
{
    // something really wrong if a single source file is more than 8 mb
    const size_t bufferSize = 1 << 23;
    if (g_scratchBuffer.size() < bufferSize) {
        g_scratchBuffer.resize(bufferSize);
    }

    std::vector<const char*> lines;
    std::vector<size_t> lengths;
    // offset in g_scratchBuffer for the next line
    size_t pos = 0;
    char* line = g_scratchBuffer.data();
    std::ifstream stream(filePath);

    // store each line as char*; clCreateProgram() wants char**
    while (stream.getline(line, bufferSize - pos)) {
        lines.push_back(line);
        size_t length = std::strlen(line);
        // getline() doesn't store the delimiter '\n'
        std::memset(line + length++, '\n', 1);
        // each line passed to clCreateProgram() must be null terminated
        std::memset(line + length, '\0', 1);
        // line size must exclude null terminator
        lengths.push_back(length);

        // + 1 to go past the null terminator
        length++;
        pos += length;
        line += length;
        if (pos >= bufferSize) {
            std::cerr << filePath << " is more than " << bufferSize << " bytes"
                      << std::endl;
            return d_ocl_manager<cl_program>::makeShared(nullptr, nullptr);
        }
    }

    std::shared_ptr<d_ocl_manager<cl_program>> program
        = d_ocl_manager<cl_program>::makeShared(
            clCreateProgramWithSource(
                context, lines.size(), lines.data(), lengths.data(), nullptr),
            &clReleaseProgram);
    if (!program) {
        return program;
    }

    // compile and link the program
    if (clBuildProgram(
            program->openclObject, 0, nullptr, nullptr, nullptr, nullptr)
        != CL_SUCCESS) {
        // clReleaseProgram if build failed
        program.reset();
    }
    return program;
}

// wrapper around clGetDeviceInfo()
template<typename T>
auto information(cl_device_id device,
                 cl_device_info param_name,
                 std::vector<T>& param_value,
                 T default_value) -> bool
{
    // first find out value string length
    size_t requiredSize = 0;
    // requiredSize will be set to value string length
    if (clGetDeviceInfo(
            device, param_name, 0, param_value.data(), &requiredSize)
            != CL_SUCCESS
        || !requiredSize) {
        return false;
    }

    // add 1 for safety, like null-termination
    param_value.resize(requiredSize + 1, default_value);
    return (clGetDeviceInfo(
                device, param_name, requiredSize, param_value.data(), nullptr)
            == CL_SUCCESS);
};

auto description(cl_device_id device) -> std::string
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
            for (int j = 0; j < value.size(); j++) {
                stream << value[j];
                if (j != 0) {
                    // space between numbers if multi-item value
                    stream << " ";
                }
            }
        }
        stream << std::endl;
    }

    return stream.str();
}

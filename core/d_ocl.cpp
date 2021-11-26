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

auto d_ocl::check_run(const std::string& funcName, cl_int funcRetval) -> bool
{
    if (funcRetval == CL_SUCCESS) {
        return true;
    }

    std::cerr << funcName << "() -> " << funcRetval << std::endl;
    return false;
}

auto d_ocl::gpuPlatforms() -> std::vector<cl_platform_id>
{
    // find available platform count first
    cl_uint numPlatforms;
    if (!d_ocl::check_run("clGetPlatformIDs",
                          clGetPlatformIDs(0, nullptr, &numPlatforms))) {
        numPlatforms = 0;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (numPlatforms == 0
        || !d_ocl::check_run(
            "clGetPlatformIDs",
            clGetPlatformIDs(numPlatforms, platforms.data(), nullptr))) {
        platforms.clear();
    }

    return platforms;
}

auto d_ocl::gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>
{
    cl_uint numDevices;
    if (!d_ocl::check_run(
            "clGetDeviceIDs",
            clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices))) {
        numDevices = 0;
    }

    std::vector<cl_device_id> devices(numDevices);
    if (numDevices == 0
        || !d_ocl::check_run("clGetDeviceIDs",
                             clGetDeviceIDs(platform,
                                            CL_DEVICE_TYPE_GPU,
                                            numDevices,
                                            devices.data(),
                                            nullptr))) {
        devices.clear();
    }

    return devices;
}

auto d_ocl::gpuPlatformDevices()
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

auto d_ocl::createContext(cl_platform_id platform,
                          const std::vector<cl_device_id>& devices)
    -> std::shared_ptr<manager<cl_context>>
{
    // list is terminated with 0. request to use this platform for the context
    std::vector<cl_context_properties> contextProperties
        = {CL_CONTEXT_PLATFORM,
           reinterpret_cast<cl_context_properties>(platform),
           0};

    return manager<cl_context>::makeShared(
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

auto d_ocl::createCmdQueue(cl_device_id device, cl_context context)
    -> std::shared_ptr<manager<cl_command_queue>>
{
    return manager<cl_command_queue>::makeShared(
        clCreateCommandQueueWithProperties(context, device, nullptr, nullptr),
        &clReleaseCommandQueue);
}

auto d_ocl::createBasicPalette(basic_palette& palette) -> bool
{
    std::unordered_map<cl_platform_id, std::vector<cl_device_id>>
        platformDevices = gpuPlatformDevices();
    if (platformDevices.empty()) {
        std::cerr << "no gpu device found" << std::endl;
        return false;
    }
    const auto platformIter = platformDevices.cbegin();
    // guaranteed to have at least 1 device in platform.
    // just going to use the first one
    cl_device_id device = platformIter->second[0];

    std::shared_ptr<manager<cl_context>> context = d_ocl::createContext(
        platformIter->first, std::vector<cl_device_id>(1, device));
    if (!context) {
        std::cerr << "error creating gpu device context" << std::endl;
        return false;
    }
    // to communicate with device
    std::shared_ptr<manager<cl_command_queue>> cmdQueue
        = d_ocl::createCmdQueue(device, context->openclObject);
    if (!cmdQueue) {
        std::cerr << "error creating gpu device cmd queue" << std::endl;
        return false;
    }

    palette.context = context;
    palette.cmdQueue = cmdQueue;
    return true;
}

auto d_ocl::createProgram(cl_context context, const std::string& filePath)
    -> std::shared_ptr<manager<cl_program>>
{
    // something really wrong if a single source file is more than 8 mb
    const size_t bufferSize = 1 << 23;
    if (g_scratchBuffer.size() < bufferSize) {
        g_scratchBuffer.resize(bufferSize);
    }
    std::memset(g_scratchBuffer.data(), '\0', bufferSize);

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
            return manager<cl_program>::makeShared(nullptr, nullptr);
        }
    }

    std::shared_ptr<manager<cl_program>> program
        = manager<cl_program>::makeShared(
            clCreateProgramWithSource(
                context, lines.size(), lines.data(), lengths.data(), nullptr),
            &clReleaseProgram);
    if (!program) {
        return program;
    }

    // compile and link the program
    if (!d_ocl::check_run("clBuildProgram",
                          clBuildProgram(program->openclObject,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr))) {
        // clReleaseProgram if build failed
        program.reset();
    }
    return program;
}

// wrapper around clGetDeviceInfo()
template<typename T>
auto d_ocl::information(cl_device_id device,
                        cl_device_info param_name,
                        std::vector<T>& param_value,
                        T default_value) -> bool
{
    // first find out value string length
    size_t requiredSize = 0;
    // requiredSize will be set to value string length
    if (!d_ocl::check_run(
            "clGetDeviceInfo",
            clGetDeviceInfo(device, param_name, 0, nullptr, &requiredSize))) {
        return false;
    }

    // add 1 for safety, like null-termination
    param_value.resize(requiredSize + 1, default_value);
    return d_ocl::check_run(
        "clGetDeviceInfo",
        clGetDeviceInfo(
            device, param_name, requiredSize, param_value.data(), nullptr));
};

auto d_ocl::description(cl_device_id device) -> std::string
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

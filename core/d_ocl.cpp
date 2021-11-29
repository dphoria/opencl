#include "d_ocl.h"
#include "d_ocl_utils.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

static std::mutex g_mutex;
// all-purpose buffer e.g. for file i/o
static std::vector<char> g_scratchBuffer;

auto d_ocl::gpuPlatforms() -> std::vector<cl_platform_id>
{
    // find available platform count first
    cl_uint numPlatforms;
    if (!d_ocl::utils::checkRun(
            std::function<cl_int(cl_uint, cl_platform_id*, cl_uint*)>(
                clGetPlatformIDs),
            (cl_uint)0,
            (cl_platform_id*)nullptr,
            &numPlatforms)) {
        numPlatforms = 0;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (numPlatforms == 0
        || !d_ocl::utils::checkRun(
            std::function<cl_int(cl_uint, cl_platform_id*, cl_uint*)>(
                clGetPlatformIDs),
            (cl_uint)numPlatforms,
            (cl_platform_id*)platforms.data(),
            (cl_uint*)nullptr)) {
        platforms.clear();
    }

    return platforms;
}

auto d_ocl::gpuDevices(cl_platform_id platform) -> std::vector<cl_device_id>
{
    cl_uint numDevices;
    if (!utils::checkRun(std::function<cl_int(cl_platform_id,
                                              cl_device_type,
                                              cl_uint,
                                              cl_device_id*,
                                              cl_uint*)>(clGetDeviceIDs),
                         (cl_platform_id)platform,
                         (cl_device_type)CL_DEVICE_TYPE_GPU,
                         (cl_uint)0,
                         (cl_device_id*)nullptr,
                         &numDevices)) {
        numDevices = 0;
    }

    std::vector<cl_device_id> devices(numDevices);
    if (numDevices == 0
        || !utils::checkRun(std::function<cl_int(cl_platform_id,
                                                 cl_device_type,
                                                 cl_uint,
                                                 cl_device_id*,
                                                 cl_uint*)>(clGetDeviceIDs),
                            (cl_platform_id)platform,
                            (cl_device_type)CL_DEVICE_TYPE_GPU,
                            (cl_uint)numDevices,
                            (cl_device_id*)devices.data(),
                            (cl_uint*)nullptr)) {
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
    -> std::shared_ptr<d_ocl::utils::manager<cl_context>>
{
    // list is terminated with 0. request to use this platform for the context
    std::vector<cl_context_properties> contextProperties
        = {CL_CONTEXT_PLATFORM,
           reinterpret_cast<cl_context_properties>(platform),
           0};

    return d_ocl::utils::manager<cl_context>::makeShared(
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
    -> std::shared_ptr<d_ocl::utils::manager<cl_command_queue>>
{
    return d_ocl::utils::manager<cl_command_queue>::makeShared(
        clCreateCommandQueueWithProperties(context, device, nullptr, nullptr),
        &clReleaseCommandQueue);
}

auto d_ocl::createContextSet(context_set& contextSet) -> bool
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

    std::shared_ptr<d_ocl::utils::manager<cl_context>> context
        = d_ocl::createContext(platformIter->first,
                               std::vector<cl_device_id>(1, device));
    if (!context) {
        std::cerr << "error creating gpu device context" << std::endl;
        return false;
    }
    // to communicate with device
    std::shared_ptr<d_ocl::utils::manager<cl_command_queue>> cmdQueue
        = d_ocl::createCmdQueue(device, context->openclObject);
    if (!cmdQueue) {
        std::cerr << "error creating gpu device cmd queue" << std::endl;
        return false;
    }

    contextSet.context = context;
    contextSet.cmdQueue = cmdQueue;
    return true;
}

auto d_ocl::createProgram(cl_context context, const std::string& filePath)
    -> std::shared_ptr<d_ocl::utils::manager<cl_program>>
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
            return d_ocl::utils::manager<cl_program>::makeShared(nullptr,
                                                                 nullptr);
        }
    }

    std::shared_ptr<d_ocl::utils::manager<cl_program>> program
        = d_ocl::utils::manager<cl_program>::makeShared(
            clCreateProgramWithSource(
                context, lines.size(), lines.data(), lengths.data(), nullptr),
            &clReleaseProgram);
    if (!program) {
        return program;
    }

    // compile and link the program
    if (!d_ocl::utils::check_run("clBuildProgram",
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

auto getImageFormat(const cv::Mat& mat, cl_image_format& imageFormat) -> bool
{
    cl_channel_type channelType;
    switch (mat.depth()) {
    case CV_8S:
        channelType = CL_SIGNED_INT8;
        break;
    case CV_8U:
        // most common
        channelType = CL_UNSIGNED_INT8;
        break;

    case CV_16F:
        channelType = CL_HALF_FLOAT;
        break;
    case CV_16S:
        channelType = CL_SIGNED_INT16;
        break;
    case CV_16U:
        channelType = CL_UNSIGNED_INT16;
        break;

    case CV_32F:
        channelType = CL_FLOAT;
        break;
    case CV_32S:
        channelType = CL_SIGNED_INT32;
        break;

    default:
        std::cerr << "unsupported cv::Mat::depth() " << mat.depth()
                  << std::endl;
        return false;
    }

    cl_channel_order channelOrder;
    switch (mat.channels()) {
    case 1:
        channelOrder = CL_R;
        break;
    case 3:
        channelOrder = CL_RGB;
        break;
    case 4:
        channelOrder = CL_RGBA;
        break;
    default:
        std::cerr << "unsupported cv::Mat.channels() " << mat.channels()
                  << std::endl;
        return false;
    }

    imageFormat.image_channel_data_type = channelType;
    imageFormat.image_channel_order = channelOrder;
    return true;
}

auto getImageDescription(const cv::Mat& mat, cl_image_desc& description) -> bool
{
    if (mat.cols <= 0 || mat.rows <= 0 || mat.cols > CL_DEVICE_IMAGE2D_MAX_WIDTH
        || mat.rows > CL_DEVICE_IMAGE2D_MAX_HEIGHT) {
        std::cerr << "invalid image resolution " << mat.cols << "x" << mat.rows
                  << std::endl;
        return false;
    }

    memset(&description, 0, sizeof(description));
    description.image_type = CL_MEM_OBJECT_IMAGE2D;
    description.image_width = mat.cols;
    description.image_height = mat.rows;
    return true;
}

auto d_ocl::createInputImage(
    cl_context context,
    cl_mem_flags flags,
    const std::string& filePath,
    const std::vector<d_ocl::utils::mat_convert_func>& matConverts,
    cv::Mat* opencvMat /*= nullptr*/
    ) -> std::shared_ptr<d_ocl::utils::manager<cl_mem>>
{
    cv::Mat srcMat = cv::imread(filePath);
    if (srcMat.empty()) {
        std::cerr << "cv::imread(" << filePath << ") failed" << std::endl;
        return std::shared_ptr<d_ocl::utils::manager<cl_mem>>();
    }

    cv::Mat finalMat = srcMat;
    // apply apply requested conversions like bgra -> rgba
    for (d_ocl::utils::mat_convert_func convertFunc : matConverts) {
        convertFunc(&srcMat, &finalMat);
        srcMat = finalMat;
    }

    // now ready to map to opencl image meta data
    cl_image_format imageFormat;
    cl_image_desc imageDesc;
    if (!getImageFormat(finalMat, imageFormat)
        || !getImageDescription(finalMat, imageDesc)) {
        return std::shared_ptr<d_ocl::utils::manager<cl_mem>>();
    }
    // input image byte size per row
    imageDesc.image_row_pitch = finalMat.step[0];

    cl_int status;
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> image
        = d_ocl::utils::manager<cl_mem>::makeShared(
            // initialize the device-side buffer with the input image
            clCreateImage(context,
                          flags | CL_MEM_COPY_HOST_PTR,
                          &imageFormat,
                          &imageDesc,
                          finalMat.data,
                          &status),
            &clReleaseMemObject);
    if (!image || status != CL_SUCCESS) {
        std::cerr << "clCreateImage() for " << filePath << " failed: " << status
                  << std::endl;
    } else if (opencvMat != nullptr) {
        *opencvMat = finalMat;
    }

    return image;
}

auto d_ocl::createOutputImage(cl_context context,
                              cl_mem_flags flags,
                              const cv::Mat& opencvMat)
    -> std::shared_ptr<d_ocl::utils::manager<cl_mem>>
{
    cl_image_format imageFormat;
    cl_image_desc imageDesc;
    if (!getImageFormat(opencvMat, imageFormat)
        || !getImageDescription(opencvMat, imageDesc)) {
        return std::shared_ptr<d_ocl::utils::manager<cl_mem>>();
    }

    cl_int status;
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> image
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateImage(
                context, flags, &imageFormat, &imageDesc, nullptr, &status),
            &clReleaseMemObject);
    if (!image || status != CL_SUCCESS) {
        std::cerr << "clCreateImage() for output image failed: " << status
                  << std::endl;
    }

    return image;
}

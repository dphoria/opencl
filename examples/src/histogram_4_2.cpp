#include "histogram_4_2.h"
#include "../../core/d_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#define EX_NAME_HISTOGRAM_4_2 "histogram_4_2"
#define EX_KERN_HISTOGRAM_4_2 histogram_4_2
// must match HIST_BINS in the opencl kernel
#define HIST_BINS 256

auto histogram_4_2() -> bool
{
    // gpu context and command queue for the first gpu device found
    d_ocl::basic_palette palette;
    if (!d_ocl::createBasicPalette(palette)) {
        std::cerr << "error creating gpu context and command queue"
                  << std::endl;
        return false;
    }

    // input image for histogram
    cv::Mat bmp = cv::imread(EX_RESOURCE_ROOT "/cat.bmp");
    if (bmp.empty()) {
        std::cerr << "did not find cat.bmp" << std::endl;
        return false;
    }
    if (bmp.depth() != CV_8U) {
        std::cerr << "input image must contain pixel data in unsigned 8-bit int"
                  << std::endl;
        return false;
    }

    // image size in bytes
    const size_t imageSize = bmp.step[0] * bmp.rows;
    // total sample count (pixels * channels per pixel)
    const size_t imageElements = bmp.rows * bmp.cols * bmp.channels();
    // host-side histogram
    std::vector<int> hostHistogram(HIST_BINS, 0);
    const size_t histogramSize = hostHistogram.size() * sizeof(int);

    // buffer object for the input image
    // initialized with pixel data from bmp
    std::shared_ptr<d_ocl::manager<cl_mem>> deviceImg
        = d_ocl::manager<cl_mem>::makeShared(
            clCreateBuffer(palette.context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                               | CL_MEM_COPY_HOST_PTR,
                           imageSize,
                           bmp.data,
                           nullptr),
            &clReleaseMemObject);
    // to get histogram from device to host accessible memory
    std::shared_ptr<d_ocl::manager<cl_mem>> deviceHistogram
        = d_ocl::manager<cl_mem>::makeShared(
            clCreateBuffer(palette.context->openclObject,
                           CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                           histogramSize,
                           nullptr,
                           nullptr),
            &clReleaseMemObject);
    if (!deviceImg || !deviceHistogram) {
        std::cerr << "error creating buffers for input image and histogram"
                  << std::endl;
        return false;
    }

    // initialize the output histogram with zeros
    const int zero = 0;
    cl_event histogramInitialized;
    if (!d_ocl::check_run("clEnqueueFillBuffer",
                          clEnqueueFillBuffer(palette.cmdQueue->openclObject,
                                              deviceHistogram->openclObject,
                                              &zero,
                                              sizeof(zero),
                                              0,
                                              histogramSize,
                                              0,
                                              nullptr,
                                              &histogramInitialized))) {
        return false;
    }

    // compile and link the program
    std::shared_ptr<d_ocl::manager<cl_program>> program = d_ocl::createProgram(
        palette.context->openclObject,
        EX_RESOURCE_ROOT "/" EX_NAME_HISTOGRAM_4_2 "." D_OCL_KERN_EXT);

    std::shared_ptr<d_ocl::manager<cl_kernel>> kernel;
    if (program) {
        // make a kernel out of the program
        kernel = d_ocl::manager<cl_kernel>::makeShared(
            // specify the kernel name, decorated with __kernel in the source
            clCreateKernel(
                program->openclObject, EX_NAME_HISTOGRAM_4_2, nullptr),
            &clReleaseKernel);
    }

    if (!program
        || !kernel
        // arguments for
        // histogram_4_2(
        //     __global unsigned char* data, int numData, __global int*
        //     histogram)
        || !d_ocl::check_run("clSetKernelArg",
                             clSetKernelArg(kernel->openclObject,
                                            0,
                                            sizeof(cl_mem),
                                            &deviceImg->openclObject))
        || !d_ocl::check_run(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 1, sizeof(imageElements), &imageElements))
        || !d_ocl::check_run("clSetKernelArg",
                             clSetKernelArg(kernel->openclObject,
                                            2,
                                            sizeof(cl_mem),
                                            &deviceHistogram->openclObject))) {
        std::cerr << "error creating program kernel" << std::endl;
        return false;
    }

    std::unordered_map<cl_platform_id, std::vector<cl_device_id>>
        platformDevices = d_ocl::gpuPlatformDevices();
    const auto platformIter = platformDevices.cbegin();
    // createBasicPalette() above just used the first gpu device for the context
    cl_device_id device = platformIter->second[0];

    // max # work-items per compute unit
    std::vector<cl_uint> maxWorkGroupSize;
    // max # work-items per dimension
    std::vector<cl_uint> maxWorkItemsByDim;
    // # parallel compute units
    // keep in mind a work-group executes on a single compute unit
    std::vector<cl_uint> numComputeUnits;
    if (!d_ocl::information<cl_uint>(
            device, CL_DEVICE_MAX_WORK_GROUP_SIZE, maxWorkGroupSize, 0)
        || !d_ocl::information<cl_uint>(
            device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemsByDim, 0)
        || !d_ocl::information<cl_uint>(
            device, CL_DEVICE_MAX_COMPUTE_UNITS, numComputeUnits, 0)) {
        std::cerr << "could not calculate appropriate execution topology"
                  << std::endl;
        return false;
    }

    size_t workGroupSize = std::min(maxWorkGroupSize[0], maxWorkItemsByDim[0]);
    // will read at most this many pixel data points concurrently
    // max # parallel compute units (work-groups)
    // * max # work-items in work-group
    size_t globalSize = numComputeUnits[0] * workGroupSize;

    std::cout << "input image: " << imageElements << " elements" << std::endl
              << "global size: " << globalSize << std::endl
              << "work-groups: "
              << std::ceil((float)imageElements / workGroupSize) << std::endl
              << "work-items per work-group (local size): " << workGroupSize
              << std::endl;

    cl_event kernel_event;
    // queue the kernel onto the device
    // read the answer into host buffer after kernel is finished
    if (!d_ocl::check_run("clEnqueueNDRangeKernel",
                          clEnqueueNDRangeKernel(palette.cmdQueue->openclObject,
                                                 kernel->openclObject,
                                                 1,
                                                 nullptr,
                                                 &globalSize,
                                                 &workGroupSize,
                                                 0,
                                                 nullptr,
                                                 &kernel_event))
        || !d_ocl::check_run("clEnqueueReadBuffer",
                             clEnqueueReadBuffer(palette.cmdQueue->openclObject,
                                                 deviceHistogram->openclObject,
                                                 CL_TRUE,
                                                 0,
                                                 histogramSize,
                                                 hostHistogram.data(),
                                                 1,
                                                 &kernel_event,
                                                 nullptr))) {
        return false;
    }

    // brute-force histogram
    std::vector<int> histogram(HIST_BINS, 0);
    // each item is a component in the given pixel
    // e.g. for RGB: {pixel 0 R, pixel 0 G, pixel 0 B, pixel 1 R, ...}
    const uint8_t* pixelData = reinterpret_cast<uint8_t*>(bmp.data);
    for (size_t pixel = 0; pixel < imageElements; pixel++) {
        histogram[pixelData[pixel]]++;
    }

    // compare answers
    for (size_t bin = 0; bin < HIST_BINS; bin++) {
        if (histogram[bin] != hostHistogram[bin]) {
            std::cerr << "opencl histogram[" << bin
                      << "] = " << hostHistogram[bin]
                      << " != " << histogram[bin] << " = c++ histogram[" << bin
                      << "]" << std::endl;
            return false;
        }
    }

    return true;
}

// append to g_exampleNames and g_exampleFunctions
D_OCL_REGISTER_EXAMPLE(EX_KERN_HISTOGRAM_4_2, EX_NAME_HISTOGRAM_4_2)
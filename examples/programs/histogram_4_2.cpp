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

    // image size in bytes
    const size_t imageSize = bmp.step[0] * bmp.rows;
    // total pixel count
    const int imageElements = bmp.rows * bmp.cols;
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

    // TODO: createProgram() fails with cl_status -11
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
        // histogram_4_2(__global int* data, int numData, __global int*
        // histogram)
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

    // # compute units == # work items == # concurrent kernel program instances
    std::vector<cl_uint> numWorkItems(1, 0);
    if (!d_ocl::information<cl_uint>(
            device, CL_DEVICE_MAX_COMPUTE_UNITS, numWorkItems, 0)
        || numWorkItems[0] == 0) {
        std::cerr << "could not query compute unit count on current gpu device"
                  << std::endl;
        return false;
    }
    // strip in image == work-group
    size_t workGroupSize = std::min<size_t>(numWorkItems[0], bmp.cols);

    cl_event kernel_event;
    // queue the kernel onto the device
    // read the answer into host buffer after kernel is finished
    if (!d_ocl::check_run(
            "clEnqueueNDRangeKernel",
            clEnqueueNDRangeKernel(palette.cmdQueue->openclObject,
                                   kernel->openclObject,
                                   1,
                                   nullptr,
                                   reinterpret_cast<size_t*>(&numWorkItems[0]),
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

    for (size_t pixel = 0; pixel < imageElements; pixel++) {
        std::cout << "histogram[" << pixel << "] = " << hostHistogram[pixel]
                  << std::endl;
    }

    return true;
}

// append to g_exampleNames and g_exampleFunctions
D_OCL_REGISTER_EXAMPLE(EX_KERN_HISTOGRAM_4_2, EX_NAME_HISTOGRAM_4_2)
#include "histogram_4_2.h"
#include "../../core/d_ocl.h"
#include "programs_defines.h"
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
    d_ocl::context_set contextSet;
    if (!d_ocl::createContextSet(contextSet)) {
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
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> deviceImg
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateBuffer(contextSet.context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                               | CL_MEM_COPY_HOST_PTR,
                           imageSize,
                           bmp.data,
                           nullptr),
            &clReleaseMemObject);
    // to get histogram from device to host accessible memory
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> deviceHistogram
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateBuffer(contextSet.context->openclObject,
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
    if (!d_ocl::utils::checkRun(
            "clEnqueueFillBuffer",
            clEnqueueFillBuffer(contextSet.cmdQueue->openclObject,
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
    std::shared_ptr<d_ocl::utils::manager<cl_program>> program
        = d_ocl::createProgram(contextSet.context->openclObject,
                               EX_RESOURCE_ROOT "/" EX_NAME_HISTOGRAM_4_2
                                                "." D_OCL_KERN_EXT);

    std::shared_ptr<d_ocl::utils::manager<cl_kernel>> kernel;
    if (program) {
        // make a kernel out of the program
        kernel = d_ocl::utils::manager<cl_kernel>::makeShared(
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
        || !d_ocl::utils::checkRun("clSetKernelArg",
                                   clSetKernelArg(kernel->openclObject,
                                                  0,
                                                  sizeof(cl_mem),
                                                  &deviceImg->openclObject))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 1, sizeof(imageElements), &imageElements))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(kernel->openclObject,
                           2,
                           sizeof(cl_mem),
                           &deviceHistogram->openclObject))) {
        std::cerr << "error creating program kernel" << std::endl;
        return false;
    }

    // max # work-items per compute unit
    std::vector<size_t> workGroupSize
        = d_ocl::utils::maxWorkGroupSize(contextSet.device);
    // # parallel compute units
    // keep in mind a work-group executes on a single compute unit
    cl_uint numComputeUnits = d_ocl::utils::maxComputeUnits(contextSet.device);
    if (numComputeUnits == 0) {
        return false;
    }

    // will read at most this many pixel data points concurrently
    // max # parallel compute units (work-groups)
    // * max # work-items in work-group
    size_t globalSize = numComputeUnits * workGroupSize[0];

    std::cout << "input image: " << imageElements << " elements" << std::endl
              << "global size: " << globalSize << std::endl
              << "work-groups: "
              << std::ceil((float)imageElements / workGroupSize[0]) << std::endl
              << "work-items per work-group (local size): " << workGroupSize[0]
              << std::endl;

    cl_event kernel_event;
    // queue the kernel onto the device
    // read the answer into host buffer after kernel is finished
    if (!d_ocl::utils::checkRun(
            "clEnqueueNDRangeKernel",
            clEnqueueNDRangeKernel(contextSet.cmdQueue->openclObject,
                                   kernel->openclObject,
                                   1,
                                   nullptr,
                                   &globalSize,
                                   workGroupSize.data(),
                                   0,
                                   nullptr,
                                   &kernel_event))
        || !d_ocl::utils::checkRun(
            "clEnqueueReadBuffer",
            clEnqueueReadBuffer(contextSet.cmdQueue->openclObject,
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
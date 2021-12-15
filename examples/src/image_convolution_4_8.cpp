#include "image_convolution_4_8.h"
#include "../../core/d_ocl.h"
#include "programs_defines.h"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define EX_NAME_IMG_CONVOLUTION_4_8 "image_convolution_4_8"
#define EX_KERN_IMG_CONVOLUTION_4_8 image_convolution_4_8

static float gaussianBlurFilterFactor = 273.0f;
static std::vector<float> gaussianBlurFilter
    = {1.0f,  4.0f, 7.0f,  4.0f,  1.0f,  4.0f, 16.0f, 26.0f, 16.0f,
       4.0f,  7.0f, 26.0f, 41.0f, 26.0f, 7.0f, 4.0f,  16.0f, 26.0f,
       16.0f, 4.0f, 1.0f,  4.0f,  7.0f,  4.0f, 1.0f};
static const int gaussianBlurFilterWidth = 5;

auto image_convolution_4_8() -> bool
{
    // context and command queue for the first gpu device found
    d_ocl::context_set contextSet;
    if (!d_ocl::createContextSet(contextSet)) {
        return false;
    }

    // read in the src image in rgba 32-bit float (format expected by kernel
    // program)
    cv::Mat inputMat;
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> inputImage
        = d_ocl::createInputImage(contextSet.context->openclObject,
                                  CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                  EX_RESOURCE_ROOT "/cat.bmp",
                                  {d_ocl::utils::toRgba, d_ocl::utils::toFloat},
                                  &inputMat);
    // deviec-side buffer for output image
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> outputImage
        = d_ocl::createOutputImage(contextSet.context->openclObject,
                                   CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                   inputMat);
    if (!inputImage || !outputImage) {
        return false;
    }

    // will pass filter coefficients into the kernel as an arg
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> filter
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateBuffer(contextSet.context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR
                               | CL_MEM_HOST_WRITE_ONLY,
                           gaussianBlurFilter.size() * sizeof(float),
                           gaussianBlurFilter.data(),
                           nullptr),
            clReleaseMemObject);
    // don't go beyond the src image
    std::vector<cl_sampler_properties> samplerProps
        = {CL_SAMPLER_NORMALIZED_COORDS,
           CL_FALSE,
           CL_SAMPLER_ADDRESSING_MODE,
           CL_ADDRESS_CLAMP_TO_EDGE,
           CL_SAMPLER_FILTER_MODE,
           CL_FILTER_NEAREST,
           0};
    std::shared_ptr<d_ocl::utils::manager<cl_sampler>> sampler
        = d_ocl::utils::manager<cl_sampler>::makeShared(
            clCreateSamplerWithProperties(
                contextSet.context->openclObject, samplerProps.data(), nullptr),
            clReleaseSampler);
    if (!filter || !sampler) {
        return false;
    }

    std::shared_ptr<d_ocl::utils::manager<cl_program>> program
        = d_ocl::createProgram(contextSet.context->openclObject,
                               EX_RESOURCE_ROOT "/" EX_NAME_IMG_CONVOLUTION_4_8
                                                "." D_OCL_KERN_EXT);
    if (!program) {
        return false;
    }
    std::shared_ptr<d_ocl::utils::manager<cl_kernel>> kernel
        = d_ocl::utils::manager<cl_kernel>::makeShared(
            clCreateKernel(
                program->openclObject, EX_NAME_IMG_CONVOLUTION_4_8, nullptr),
            clReleaseKernel);
    // args for
    // void image_convolution_4_8(
    //                        int imageWidth,
    //                        int imageHeight,
    //      __read_only image2d_t inputImage,
    //     __write_only image2d_t outputImage,
    //          __constant float* filter,
    //                        int filterWidth,
    //                  sampler_t sampler)
    if (!kernel
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 0, sizeof(inputMat.cols), &inputMat.cols))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 1, sizeof(inputMat.rows), &inputMat.rows))
        || !d_ocl::utils::checkRun("clSetKernelArg",
                                   clSetKernelArg(kernel->openclObject,
                                                  2,
                                                  sizeof(cl_mem),
                                                  &inputImage->openclObject))
        || !d_ocl::utils::checkRun("clSetKernelArg",
                                   clSetKernelArg(kernel->openclObject,
                                                  3,
                                                  sizeof(cl_mem),
                                                  &outputImage->openclObject))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 4, sizeof(cl_mem), &filter->openclObject))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(kernel->openclObject,
                           5,
                           sizeof(gaussianBlurFilterWidth),
                           &gaussianBlurFilterWidth))
        || !d_ocl::utils::checkRun("clSetKernelArg",
                                   clSetKernelArg(kernel->openclObject,
                                                  6,
                                                  sizeof(cl_sampler),
                                                  &sampler->openclObject))) {
        return false;
    }

    // # work-groups * # work-items = # concurrent global work-items
    cl_uint numWorkgroups = d_ocl::utils::maxComputeUnits(contextSet.device);
    std::vector<size_t> workgroupSize
        = d_ocl::utils::maxWorkGroupSize(contextSet.device);
    if (numWorkgroups == 0 || workgroupSize.size() < 2) {
        std::cerr << "unable to determine work-items topology" << std::endl;
        return false;
    }
    // 1 work-item : 1 pixel
    std::vector<size_t> globalWorkSize
        = {numWorkgroups * workgroupSize[0], numWorkgroups * workgroupSize[1]};
    cl_event kernelEvent;
    // run the image convolution
    if (!d_ocl::utils::checkRun(
            "clEnqueueNDRangeKernel",
            clEnqueueNDRangeKernel(contextSet.cmdQueue->openclObject,
                                   kernel->openclObject,
                                   2,
                                   nullptr,
                                   globalWorkSize.data(),
                                   nullptr,
                                   0,
                                   nullptr,
                                   &kernelEvent))) {
        return false;
    }

    // transfer the image to host-side
    std::vector<size_t> origin(3, 0);
    std::vector<size_t> region
        = {(size_t)inputMat.cols, (size_t)inputMat.rows, 1};
    // allocate image so we can transfer rotated image to this buffer
    cv::Mat outputMat = cv::Mat::zeros(inputMat.size(), inputMat.type());
    if (!d_ocl::utils::checkRun(
            "clEnqueueReadImage",
            clEnqueueReadImage(contextSet.cmdQueue->openclObject,
                               outputImage->openclObject,
                               CL_TRUE,
                               origin.data(),
                               region.data(),
                               inputMat.step[0],
                               0,
                               outputMat.data,
                               1,
                               &kernelEvent,
                               nullptr))) {
        return false;
    }

    // outputMat is CV_32FC4 RGBA. convert to opencv native BGR (CV_32FC3)
    // before calling cv::imwrite()
    cv::Mat bgraMat = cv::Mat(outputMat.size(), outputMat.type());
    cv::cvtColor(outputMat, bgraMat, cv::COLOR_RGBA2BGR);
    if (!cv::imwrite(EX_NAME_IMG_CONVOLUTION_4_8 ".tiff", bgraMat)) {
        std::cerr << "error saving filtered image to disk" << std::endl;
        return false;
    }

    // TODO: filtering did not work; output image is not the expected blurred image
    //       the filtering may need modifications to use just the first channel
    std::cout << "filtered image saved in " EX_NAME_IMG_CONVOLUTION_4_8 ".tiff"
              << std::endl;

    return true;
}

D_OCL_REGISTER_EXAMPLE(EX_KERN_IMG_CONVOLUTION_4_8,
                       EX_NAME_IMG_CONVOLUTION_4_8);
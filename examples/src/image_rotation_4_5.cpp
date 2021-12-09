#include "image_rotation_4_5.h"
#include "../../core/d_ocl.h"
#include "programs_defines.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define EX_NAME_IMG_ROTATION_4_5 "image_rotation_4_5"
#define EX_KERN_IMG_ROTATION_4_5 image_rotation_4_5

auto image_rotation_4_5() -> bool
{
    // gpu context and command queue for the first gpu device found
    d_ocl::context_set contextSet;
    if (!d_ocl::createContextSet(contextSet)) {
        return false;
    }

    std::string inputImagePath = EX_RESOURCE_ROOT "/cat-face.bmp";
    cv::Mat inputMat;
    // read the input image with pixel datain 32-bit floats
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> inputImage
        = d_ocl::createInputImage(
            contextSet.context->openclObject,
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
            inputImagePath,
            // the kernel expects pixel data in 32-bit floats
            {&d_ocl::utils::toRgba, &d_ocl::utils::toFloat},
            // get the input image as opencv matrix
            &inputMat);
    if (!inputImage) {
        std::cerr << "error preparing input cl_mem from " << inputImagePath
                  << std::endl;
        return false;
    }

    // device-side buffer to get the rotated image
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> outputImage
        = d_ocl::createOutputImage(
            // rotated image will have same spec as input image,
            // so tell the code to use inputMat to get properties like size
            contextSet.context->openclObject,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
            inputMat);
    if (!outputImage) {
        std::cerr << "error preparing output cl_mem " << std::endl;
    }

    std::shared_ptr<d_ocl::utils::manager<cl_program>> program
        = d_ocl::createProgram(contextSet.context->openclObject,
                               EX_RESOURCE_ROOT "/" EX_NAME_IMG_ROTATION_4_5
                                                "." D_OCL_KERN_EXT);
    if (!program) {
        return false;
    }
    std::shared_ptr<d_ocl::utils::manager<cl_kernel>> kernel
        = d_ocl::utils::manager<cl_kernel>::makeShared(
            clCreateKernel(
                program->openclObject, EX_NAME_IMG_ROTATION_4_5, nullptr),
            clReleaseKernel);

    // arbitrary image rotation angle
    float theta = 45;
    // set up args for
    // void rotation(__read_only image2d_t inputImage,
    //              __write_only image2d_t outputImage,
    //                                 int imageWidth,
    //                                 int imageHeight,
    //                               float theta)
    if (!kernel
        || !d_ocl::utils::checkRun("clSetKernelArg",
                                   clSetKernelArg(kernel->openclObject,
                                                  0,
                                                  sizeof(cl_mem),
                                                  &inputImage->openclObject))
        || !d_ocl::utils::checkRun("clSetKernelArg",
                                   clSetKernelArg(kernel->openclObject,
                                                  1,
                                                  sizeof(cl_mem),
                                                  &outputImage->openclObject))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 2, sizeof(inputMat.cols), &inputMat.cols))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(
                kernel->openclObject, 3, sizeof(inputMat.rows), &inputMat.rows))
        || !d_ocl::utils::checkRun(
            "clSetKernelArg",
            clSetKernelArg(kernel->openclObject, 4, sizeof(theta), &theta))) {
        std::cerr << "error creating and setting up kernel program"
                  << std::endl;
        return false;
    }

    // max # work-items per work-group in each dimension
    std::vector<size_t> workGroupSize
        = d_ocl::utils::maxWorkGroupSize(contextSet.device);
    // max # work-groups
    cl_uint numWorkGroups = d_ocl::utils::maxComputeUnits(contextSet.device);
    if (workGroupSize.size() < 2 || numWorkGroups == 0) {
        std::cerr << "unable to determine work-items topology" << std::endl;
        return false;
    }

    // 1:1 between pixel and work-item, i.e. exactly 1 work-item rotates exactly
    // 1 pixel
    std::vector<size_t> globalSize
        = {std::min<size_t>(workGroupSize[0] * numWorkGroups, inputMat.cols),
           std::min<size_t>(workGroupSize[1] * numWorkGroups, inputMat.rows)};

    std::cout << "input image " << inputMat.cols << "x" << inputMat.rows << ". "
              << globalSize[0] << "x" << globalSize[1]
              << " pixels rotated at once by " << theta << " degrees"
              << std::endl;

    cl_event kernelEvent;
    // queue the kernel onto the gpu
    if (!d_ocl::utils::checkRun(
            "clEnqueueNDRangeKernel",
            clEnqueueNDRangeKernel(contextSet.cmdQueue->openclObject,
                                   kernel->openclObject,
                                   2,
                                   nullptr,
                                   globalSize.data(),
                                   nullptr,
                                   0,
                                   nullptr,
                                   &kernelEvent))) {
        return false;
    }

    // transfer the rotated image to host-side and write out to local disk
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
    if (!cv::imwrite(EX_NAME_IMG_ROTATION_4_5 ".tiff", bgraMat)) {
        std::cerr << "error saving rotated image to disk" << std::endl;
        return false;
    }

    std::cout << "rotated image saved in " EX_NAME_IMG_ROTATION_4_5 ".tiff"
              << std::endl;
    return true;
}

D_OCL_REGISTER_EXAMPLE(EX_KERN_IMG_ROTATION_4_5, EX_NAME_IMG_ROTATION_4_5)
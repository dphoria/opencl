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

    // TODO: queue/execute the kernel, get the rotated image from outputImage
    // and save to local disk file

    return true;
}

D_OCL_REGISTER_EXAMPLE(EX_KERN_IMG_ROTATION_4_5, EX_NAME_IMG_ROTATION_4_5)
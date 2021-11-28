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
    d_ocl::basic_palette palette;
    if (!d_ocl::createBasicPalette(palette)) {
        return false;
    }

    std::string inputImagePath = EX_RESOURCE_ROOT "/cat-face.bmp";
    cv::Mat inputMat;
    // read the input image with pixel datain 32-bit floats
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> inputImage
        = d_ocl::createInputImage(
            palette.context->openclObject,
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

    return true;
}

D_OCL_REGISTER_EXAMPLE(EX_KERN_IMG_ROTATION_4_5, EX_NAME_IMG_ROTATION_4_5)
#include "image_convolution_4_8.h"
#include "../../core/d_ocl.h"
#include "programs_defines.h"
#include <opencv2/core.hpp>
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

    return true;
}

D_OCL_REGISTER_EXAMPLE(EX_KERN_IMG_CONVOLUTION_4_8,
                       EX_NAME_IMG_CONVOLUTION_4_8);
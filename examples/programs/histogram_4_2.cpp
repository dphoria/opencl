#include "histogram_4_2.h"
#include <CL/cl.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#define EX_NAME_HISTOGRAM_4_2 "histogram_4_2"
#define EX_KERN_HISTOGRAM_4_2 histogram_4_2
// must match HIST_BINS in the opencl kernel
#define HIST_BINS 256

auto histogram_4_2() -> bool
{
    /*
    cv::Mat bmp = cv::imread(EX_RESOURCE_ROOT "/cat.bmp");
    if (bmp.empty()) {
        std::cerr << "did not find cat.bmp" << std::endl;
        return false;
    }

    const int imageElements = bmp.rows * bmp.cols;

    // histogram on host memory
    std::vector<int> hostHistogram(HIST_BINS, 0);
    */
    std::cout << EX_NAME_HISTOGRAM_4_2 << "not implemented" << std::endl;
    return true;
}

// append to g_exampleNames and g_exampleFunctions
D_OCL_REGISTER_EXAMPLE(EX_KERN_HISTOGRAM_4_2, EX_NAME_HISTOGRAM_4_2)
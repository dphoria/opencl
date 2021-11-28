#include "vector_add_3_4.h"
#include "../../core/d_ocl.h"
#include "programs_defines.h"
#include <iostream>
#include <random>
#include <vector>

#define EX_NAME_VECTOR_ADD_3_4 "vector_add_3_4"
#define EX_KERN_VECTOR_ADD_3_4 vector_add_3_4

auto vector_add_3_4() -> bool
{
    // gpu context and command queue for the first gpu device found
    d_ocl::basic_palette palette;
    if (!d_ocl::createBasicPalette(palette)) {
        return false;
    }

    // number of items in each array
    const size_t numElements = 2048;
    // data size in bytes
    const size_t dataSize = sizeof(int) * numElements;

    // host buffers
    // input
    std::vector<int> hostA(numElements);
    std::vector<int> hostB(numElements);
    // output
    std::vector<int> hostC(numElements);

    // input data init
    std::random_device randDevice;
    std::default_random_engine randEngine(randDevice());
    std::uniform_int_distribution<int> randDistribution(
        std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    for (int i = 0; i < numElements; i++) {
        hostA[i] = randDistribution(randEngine);
        hostB[i] = randDistribution(randEngine);
    }

    // device-side memory
    // initialize with data from host-side vector
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> deviceA
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateBuffer(palette.context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                               | CL_MEM_COPY_HOST_PTR,
                           dataSize,
                           hostA.data(),
                           nullptr),
            &clReleaseMemObject);
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> deviceB
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateBuffer(palette.context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                               | CL_MEM_COPY_HOST_PTR,
                           dataSize,
                           hostB.data(),
                           nullptr),
            &clReleaseMemObject);
    // to get answer from device to host accessible memory
    std::shared_ptr<d_ocl::utils::manager<cl_mem>> deviceC
        = d_ocl::utils::manager<cl_mem>::makeShared(
            clCreateBuffer(palette.context->openclObject,
                           CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                           dataSize,
                           nullptr,
                           nullptr),
            &clReleaseMemObject);
    if (!deviceA || !deviceB || !deviceC) {
        std::cerr << "error creating device-side buffer" << std::endl;
        return false;
    }

    // compile and link the program
    std::shared_ptr<d_ocl::utils::manager<cl_program>> program
        = d_ocl::createProgram(palette.context->openclObject,
                               EX_RESOURCE_ROOT "/" EX_NAME_VECTOR_ADD_3_4
                                                "." D_OCL_KERN_EXT);

    std::shared_ptr<d_ocl::utils::manager<cl_kernel>> kernel;
    if (program) {
        // make a kernel out of the program
        kernel = d_ocl::utils::manager<cl_kernel>::makeShared(
            // specify the kernel name, decorated with __kernel in the source
            clCreateKernel(
                program->openclObject, EX_NAME_VECTOR_ADD_3_4, nullptr),
            &clReleaseKernel);
    }

    if (!program
        || !kernel
        // arguments for
        // vector_add(__global int* A, __global int* B, __global int* C)
        || !d_ocl::utils::check_run("clSetKernelArg",
                                    clSetKernelArg(kernel->openclObject,
                                                   0,
                                                   sizeof(cl_mem),
                                                   &deviceA->openclObject))
        || !d_ocl::utils::check_run("clSetKernelArg",
                                    clSetKernelArg(kernel->openclObject,
                                                   1,
                                                   sizeof(cl_mem),
                                                   &deviceB->openclObject))
        || !d_ocl::utils::check_run("clSetKernelArg",
                                    clSetKernelArg(kernel->openclObject,
                                                   2,
                                                   sizeof(cl_mem),
                                                   &deviceC->openclObject))) {
        std::cerr << "error creating program kernel" << std::endl;
        return false;
    }

    cl_event kernel_event;
    // queue the kernel onto the device
    // read the answer into host buffer after kernel is finished
    if (!d_ocl::utils::check_run(
            "clEnqueueNDRangeKernel",
            clEnqueueNDRangeKernel(palette.cmdQueue->openclObject,
                                   kernel->openclObject,
                                   1,
                                   nullptr,
                                   &numElements,
                                   nullptr,
                                   0,
                                   nullptr,
                                   &kernel_event))
        || !d_ocl::utils::check_run(
            "clEnqueueReadBuffer",
            clEnqueueReadBuffer(palette.cmdQueue->openclObject,
                                deviceC->openclObject,
                                CL_TRUE,
                                0,
                                dataSize,
                                hostC.data(),
                                1,
                                &kernel_event,
                                nullptr))) {
        return false;
    }

    // check answer
    for (size_t i = 0; i < numElements; i++) {
        if (hostA[i] + hostB[i] != hostC[i]) {
            std::cerr << hostA[i] << " + " << hostB[i] << " != " << hostC[i]
                      << std::endl;
            return false;
        }
    }

    return true;
}

// append to g_exampleNames and g_exampleFunctions
D_OCL_REGISTER_EXAMPLE(EX_KERN_VECTOR_ADD_3_4, EX_NAME_VECTOR_ADD_3_4)
#include "3.4-vector-add.h"
#include "../../core/d_ocl.h"
#include "programs_defines.h"
#include <iostream>
#include <vector>

auto vector_add_3_4() -> bool
{
    // number of items in each array
    const size_t numElements = 2048;
    // data size in bytes
    size_t dataSize = sizeof(int) * numElements;

    // host buffers
    // input
    std::vector<int> hostA(numElements);
    std::vector<int> hostB(numElements);
    // output
    std::vector<int> hostC(numElements);

    // input data init
    for (int i = 0; i < hostA.size(); i++) {
        hostA[i] = hostB[i] = i;
    }

    std::unordered_map<cl_platform_id, std::vector<cl_device_id>>
        platformDevices = gpuPlatformDevices();
    if (platformDevices.empty()) {
        std::cerr << "no gpu device found" << std::endl;
        return false;
    }
    const auto platformIter = platformDevices.cbegin();
    // guaranteed to have at least 1 device in platform.
    // just going to use the first one
    cl_device_id device = platformIter->second[0];

    std::shared_ptr<d_ocl_manager<cl_context>> context = createContext(
        platformIter->first, std::vector<cl_device_id>(1, device));
    if (!context) {
        std::cerr << "error creating gpu device context" << std::endl;
        return false;
    }
    // to communicate with device
    std::shared_ptr<d_ocl_manager<cl_command_queue>> cmdQueue
        = createCmdQueue(device, context->openclObject);
    if (!cmdQueue) {
        std::cerr << "error creating gpu device cmd queue" << std::endl;
        return false;
    }

    // device-side memory
    // initialize with data from host-side vector
    std::shared_ptr<d_ocl_manager<cl_mem>> deviceA
        = d_ocl_manager<cl_mem>::makeShared(
            clCreateBuffer(context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                               | CL_MEM_COPY_HOST_PTR,
                           dataSize,
                           hostA.data(),
                           nullptr),
            &clReleaseMemObject);
    std::shared_ptr<d_ocl_manager<cl_mem>> deviceB
        = d_ocl_manager<cl_mem>::makeShared(
            clCreateBuffer(context->openclObject,
                           CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                               | CL_MEM_COPY_HOST_PTR,
                           dataSize,
                           hostB.data(),
                           nullptr),
            &clReleaseMemObject);
    // to get answer from device to host accessible memory
    std::shared_ptr<d_ocl_manager<cl_mem>> deviceC
        = d_ocl_manager<cl_mem>::makeShared(
            clCreateBuffer(context->openclObject,
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
    std::shared_ptr<d_ocl_manager<cl_program>> program = createProgram(
        context->openclObject, PROGRAM_SRC_ROOT "/" PROG_VECTOR_ADD_3_4 ".c");
    std::shared_ptr<d_ocl_manager<cl_kernel>> kernel;
    if (program) {
        // make a kernel out of the program
        kernel = std::make_shared<d_ocl_manager<cl_kernel>>(
            clCreateKernel(program->openclObject, PROG_VECTOR_ADD_3_4, nullptr),
            &clReleaseKernel);
    }

    if (!program
        || !kernel
        // arguments for
        // vector_add(__global int* A, __global int* B, __global int* C)
        || clSetKernelArg(
               kernel->openclObject, 0, sizeof(cl_mem), &deviceA->openclObject)
               != CL_SUCCESS
        || clSetKernelArg(
               kernel->openclObject, 1, sizeof(cl_mem), &deviceB->openclObject)
               != CL_SUCCESS
        || clSetKernelArg(
               kernel->openclObject, 2, sizeof(cl_mem), &deviceC->openclObject)
               != CL_SUCCESS) {
        std::cerr << "error creating program kernel" << std::endl;
        return false;
    }

    cl_event kernel_event;
    // queue the kernel onto the device
    // read the answer into host buffer after kernel is finished
    if (clEnqueueNDRangeKernel(cmdQueue->openclObject,
                               kernel->openclObject,
                               1,
                               nullptr,
                               &numElements,
                               nullptr,
                               0,
                               nullptr,
                               &kernel_event)
            != CL_SUCCESS
        || clEnqueueReadBuffer(cmdQueue->openclObject,
                               deviceC->openclObject,
                               CL_TRUE,
                               0,
                               dataSize,
                               hostC.data(),
                               1,
                               &kernel_event,
                               nullptr)
               != CL_SUCCESS) {
        std::cerr << "error running program kernel" << std::endl;
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

// append to g_testNames and g_testFunctions
REGISTER_TEST_PROG(vector_add_3_4, PROG_VECTOR_ADD_3_4, &vector_add_3_4)
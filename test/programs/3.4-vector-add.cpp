#include "3.4-vector-add.h"
#include "../../platform/d_ocl_platform.h"
#include <iostream>
#include <vector>

auto vector_add_3_4() -> bool
{
    // number of items in each array
    const int numElements = 2048;
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
    cl_context context = createContext(platformIter->first,
                                       std::vector<cl_device_id>(1, device));
    if (context == nullptr) {
        std::cerr << "error creating gpu device context" << std::endl;
        return false;
    }
    // to communicate with device
    cl_command_queue cmdQueue = createCmdQueue(device, context);
    if (cmdQueue == nullptr) {
        std::cerr << "error creating gpu device cmd queue" << std::endl;
        clReleaseContext(context);
        return false;
    }

    // device-side memory
    // initialize with data from host-side vector
    cl_mem deviceA = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                        | CL_MEM_COPY_HOST_PTR,
                                    dataSize,
                                    hostA.data(),
                                    nullptr);
    cl_mem deviceB = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                        | CL_MEM_COPY_HOST_PTR,
                                    dataSize,
                                    hostB.data(),
                                    nullptr);
    // to get answer from device to host accessible memory
    cl_mem deviceC = clCreateBuffer(context,
                                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                    dataSize,
                                    nullptr,
                                    nullptr);
    if (deviceA == nullptr || deviceB == nullptr || deviceC == nullptr) {
        clReleaseContext(context);
        std::cerr << "error creating device-side buffer" << std::endl;
        return false;
    }

    // TODO: queue command to set up buffer
    // read the kernel source from *.c file
    // run kernel on device
    // get answer from device to host memory

    clReleaseContext(context);
    return true;
}

#include "../platform/d_ocl_platform.h"
#include <iostream>


auto main(int argc, char** argv) -> int
{
    int retval = 0;

    std::unordered_map<cl_platform_id, std::vector<cl_device_id>> platformDevices = gpuPlatformDevices();
    if (platformDevices.empty()) {
        std::cerr << "no gpu platforms / devices found" << std::endl;
    } else {
        for (const std::pair<cl_platform_id, std::vector<cl_device_id>>& iter : platformDevices) {
            for (size_t i = 0; i < iter.second.size(); i++) {
                // pretty print some information about this gpu device
                std::cout << "----" << std::endl
                          << "device " << i << ":" << std::endl
                          << "----" << std::endl
                          << description(iter.second[i]) << std::endl;
            }
        }
    }

    return retval;
}

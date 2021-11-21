#include "../core/d_ocl.h"
#include "../examples/programs/programs_defines.h"
#include <iostream>

auto main(int argc, char** argv) -> int
{
    std::unordered_map<cl_platform_id, std::vector<cl_device_id>>
        platformDevices = gpuPlatformDevices();
    if (platformDevices.empty()) {
        std::cerr << "no gpu platforms / devices found" << std::endl;
        return 1;
    }

    for (const std::pair<cl_platform_id, std::vector<cl_device_id>>& iter :
         platformDevices) {
        for (size_t i = 0; i < iter.second.size(); i++) {
            // pretty print some information about this gpu device
            std::cout << "----" << std::endl
                      << "device " << i << ":" << std::endl
                      << "----" << std::endl
                      << description(iter.second[i]) << std::endl;
        }
    }

    // ----
    // 3.4-vector-add : begin
    // ----
    // ...
    // 3.4-vector-add : fail
    // ----

    auto nameIter = g_testNames.cbegin();
    auto funcIter = g_testFunctions.cbegin();
    while (nameIter != g_testNames.cend()
           && funcIter != g_testFunctions.cend()) {
        std::cout << "----" << std::endl
                  << *nameIter << " : begin " << std::endl
                  << "----" << std::endl;
        // e.g. vector_add_3_4()
        if ((*funcIter)()) {
            std::cout << *nameIter << " : pass" << std::endl;
        } else {
            std::cerr << *nameIter << " : fail" << std::endl;
        }
        std::cout << "----" << std::endl;

        nameIter++;
        funcIter++;
    }

    return 0;
}

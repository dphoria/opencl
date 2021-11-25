#ifndef D_OCL_DEFINES_H
#define D_OCL_DEFINES_H

#include <CL/cl.h>
#include <memory>

#ifdef EXPORT_D_OCL_CORE
#define D_OCL_API __attribute__((visibility("default")))
#else
#define D_OCL_API
#endif

// file extension for operncl kernel source files
#define D_OCL_KERN_EXT "cl"

// ensure release when finished with open resource like cl_context
template<typename T>
struct d_ocl_manager
{
    // e.g. clReleaseContext
    using opencl_release_func = cl_int (*)(T);

    static auto makeShared(T openclObject, opencl_release_func releaseFunc)
        -> std::shared_ptr<d_ocl_manager<T>>
    {
        if (openclObject == nullptr) {
            // operator bool will fail test
            return std::shared_ptr<d_ocl_manager<T>>();
        }

        return std::make_shared<d_ocl_manager<T>>(openclObject, releaseFunc);
    }

    d_ocl_manager(T openclObject, opencl_release_func releaseFunc)
    {
        this->openclObject = openclObject;
        this->releaseFunc = releaseFunc;
    }
    ~d_ocl_manager()
    {
        if (openclObject != nullptr && releaseFunc != nullptr) {
            releaseFunc(openclObject);
        }
    }

    T openclObject{nullptr};
    opencl_release_func releaseFunc{nullptr};
};

#endif // D_OCL_DEFINES_H

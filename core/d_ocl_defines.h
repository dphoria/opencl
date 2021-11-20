#ifndef D_OCL_PLATFORM_DEFS_H
#define D_OCL_PLATFORM_DEFS_H


#include <CL/cl.h>

#ifdef EXPORT_D_OCL_PLATFORM
#define D_OCL_API __attribute__((visibility("default")))
#else
#define D_OCL_API
#endif

// managed to ensure release
struct d_ocl_context
{
    d_ocl_context(cl_context c)
    {
        context = c;
    }
    ~d_ocl_context()
    {
        if (context != nullptr) {
            clReleaseContext(context);
        }
    }
    cl_context context{nullptr};
};

#endif  // D_OCL_PLATFORM_DEFS_H

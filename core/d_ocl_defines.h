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

#endif // D_OCL_DEFINES_H

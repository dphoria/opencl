#ifndef D_OCL_PLATFORM_DEFS_H
#define D_OCL_PLATFORM_DEFS_H


#ifdef EXPORT_D_OCL_PLATFORM
#define D_OCL_PLATFORM_API __attribute__((visibility("default")))
#else
#define D_OCL_PLATFORM_API
#endif

#endif  // D_OCL_PLATFORM_DEFS_H

#ifndef D_OCL_EXAMPLES_DEFINES_H_
#define D_OCL_EXAMPLES_DEFINES_H_

#ifdef EXPORT_D_OCL_EXAMPLES
#define D_OCL_EXAMPLES_API __attribute__((visibility("default")))
#else
#define D_OCL_EXAMPLES_API
#endif

#endif
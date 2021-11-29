#ifndef D_OCL_EXAMPLES_DEFINES_H_
#define D_OCL_EXAMPLES_DEFINES_H_

#include <functional>
#include <list>
#include <string>

#ifdef EXPORT_D_OCL_EXAMPLES
#define D_OCL_EXAMPLES_API __attribute__((visibility("default")))
#else
#define D_OCL_EXAMPLES_API
#endif

extern D_OCL_EXAMPLES_API std::list<std::string> g_exampleNames;
using d_ocl_test_func = std::function<bool()>;
extern D_OCL_EXAMPLES_API std::list<d_ocl_test_func> g_exampleFunctions;

#endif
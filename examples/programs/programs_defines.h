#ifndef PROGRAMS_DEFINES_H
#define PROGRAMS_DEFINES_H

#include "../d_ocl_examples_defines.h"
#include <list>
#include <string>

#define PROG_VECTOR_ADD_3_4 "3.4-vector-add"

extern D_OCL_EXAMPLES_API std::list<std::string> g_testNames;
using d_ocl_test_func = bool (*)();
extern D_OCL_EXAMPLES_API std::list<d_ocl_test_func> g_testFunctions;

// REGISTER_TEST_PROG(vector_add_3_4, PROG_VECTOR_ADD_3_4, &vector_add_3_4)
#define REGISTER_TEST_PROG(a, b, c)                                            \
    static struct Register_##a                                                 \
    {                                                                          \
        Register_##a()                                                         \
        {                                                                      \
            g_testNames.emplace_back(b);                                       \
            g_testFunctions.emplace_back(c);                                   \
        }                                                                      \
    } _Register_##a_;

#endif
#ifndef PROGRAMS_DEFINES_H
#define PROGRAMS_DEFINES_H

#include "../d_ocl_examples_defines.h"

#define PROG_VECTOR_ADD_3_4 "3.4_vector_add"

// REGISTER_TEST_PROG(vector_add_3_4, PROG_VECTOR_ADD_3_4, &vector_add_3_4)
#define REGISTER_TEST_PROG(a, b, c)                                            \
    static struct Register_##a                                                 \
    {                                                                          \
        Register_##a()                                                         \
        {                                                                      \
            g_exampleNames.emplace_back(b);                                    \
            g_exampleFunctions.emplace_back(c);                                \
        }                                                                      \
    } _Register_##a_;

#endif
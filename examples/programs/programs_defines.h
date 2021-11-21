#ifndef PROGRAMS_DEFINES_H
#define PROGRAMS_DEFINES_H

#include "../d_ocl_examples.h"

#define EX_NAME_VECTOR_ADD_3_4 "vector_add_3_4"
#define EX_KERN_VECTOR_ADD_3_4 vector_add_3_4

// D_OCL_REGISTER_EXAMPLE(EX_KERN_VECTOR_ADD_3_4, EX_NAME_VECTOR_ADD_3_4)
#define D_OCL_REGISTER_EXAMPLE(a, b)                                           \
    static struct Register_##a                                                 \
    {                                                                          \
        Register_##a()                                                         \
        {                                                                      \
            g_exampleFunctions.emplace_back(&a);                               \
            g_exampleNames.emplace_back(b);                                    \
        }                                                                      \
    } _Register_##a;

#endif
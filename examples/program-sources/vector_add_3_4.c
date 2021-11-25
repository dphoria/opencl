__kernel
void vector_add_3_4(__global int* A, __global int* B, __global int* C)
{
    /* partitioned such that work item id = vector index */
    int idx = get_global_id(0);
    /* store computation result in output vector C */
    C[idx] = A[idx] + B[idx];
}

/* must match HIST_BINS in the test driver c++ function */
#define HIST_BINS   256

__kernel
void histogram_4_2(__global int* data, int numData, __global int* histogram)
{
    /*
    each work-group operates on its own histogram
    of same size as the global one
    */
    __local int localHistogram[HIST_BINS];
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    /* initialize work-group local histogram */
    for (int i = localId; i < HIST_BINS; i += get_local_size(0)) {
        localHistogram[i] = 0;
    }

    /* wait until every bin in local histogram has been initialized */
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    /*
    globalId is unique across all work-items
    i.e. each item in data is counted exactly once
    */
    for (int i = globalId; i < numData; i += get_global_size(0)) {
        /*
        atomic operations on local memory are efficient,
        relative to atomic operations on global memory

        increment the data[i]-th bin by 1
        */
        atomic_store(localHistogram + data[i], atomic_load(localHistogram + data[i]) + 1);
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    /* add each work-group's local histogram to global */
    for (int i = localId; i < HIST_BINS; i += get_local_size(0)) {
        atomic_store(data + i, atomic_load(data + i) + atomic_load(localHistogram + i));
    }
}

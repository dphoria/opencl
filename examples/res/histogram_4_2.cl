/* must match HIST_BINS in the test driver c++ function */
#define HIST_BINS   256

__kernel
void histogram_4_2(__global unsigned char* data, int numData, __global int* histogram)
{
    /* each work-group operates on its own histogram */
    /* of same size as the global one */
    __local int localHistogram[HIST_BINS];
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    /* initialize work-group local histogram */
    /* step by local_size (work-group size) so this code will work even if local_size != HIST_BINS */
    for (int i = localId; i < HIST_BINS; i += get_local_size(0)) {
        /* localId is unique within work-group {0, 1, ..., local_size - 1} */
        /* so this line can execute concurrently by the work-items */
        localHistogram[i] = 0;
    }

    /* wait until every bin in local histogram has been initialized */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* data is unsigned char*, therefore i is like */
    /* {pixel 0 channel B, pixel 0 channel G, pixel 0 channel R, pixel 1 channel B, ...} */
    for (int i = globalId; i < numData; i += get_global_size(0)) {
        atomic_add(localHistogram + data[i], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* add each work-group's local histogram to global */
    for (int i = localId; i < HIST_BINS; i += get_local_size(0)) {
        atomic_add(histogram + i, localHistogram[i]);
    }

    /* reduced atomic operations on global memory from # pixels to # bins */
}

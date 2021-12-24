__kernel
void image_convolution_4_8(
                       int imageWidth,
                       int imageHeight,
     __read_only image2d_t inputImage,
    __write_only image2d_t outputImage,
         __constant float* filter,
                       int filterWidth,
                 sampler_t sampler)
{
    /* Half the width of the filter is needed for indexing
     * memory later */
    int halfWidth = filterWidth / 2.0;

    /* Store each work-item’s unique row and column */
    for (int column = get_global_id(0); column < imageWidth; column += get_global_size(0)) {
        for (int row = get_global_id(1); row < imageHeight; row += get_global_size(1)) {
            /* All accesses to images return data as four-element vector */
            float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
            /* Iterator for the filter */
            int filterIdx = 0;
            /* Each work-item iterates around its local area based on the
             * size of the filter */
            int2 coords; // Coordinates for accessing the imag

            /* Iterate the filter rows */
            for (int i = -halfWidth; i <= halfWidth; i++) {
                coords.y = row + i;
                /* Iterate over the filter columns */
                for (int j = -halfWidth; j <= halfWidth; j++) {
                    coords.x = column + j;
                    /* Read a pixel from the image */
                    float4 pixel = read_imagef(inputImage, sampler, coords);

                    /* add filtered to the new pixel data */
                    sum.x += pixel.x * filter[filterIdx++];
                }
            }

            /* Copy the data to the output image */
            coords.x = column;
            coords.y = row;
            /* only this work-item writes to this particular pixel */
            /* because we are using global id for the coordinates */
            write_imagef(outputImage, coords, sum);
        }
    }
}

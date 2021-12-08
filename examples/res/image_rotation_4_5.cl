                               // coordinates are [0...size), not [0...1]
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
                               // linear interpolation when reading from between pixels
                               // can use because we will pass coordinates in float
                               | CLK_FILTER_LINEAR
                               // prevent reading from out-of-bound pixel
                               | CLK_ADDRESS_CLAMP;

__kernel 
void image_rotation_4_5(
    // expected to store data in 32-bit float
    __read_only image2d_t inputImage, 
   __write_only image2d_t outputImage,
                      int imageWidth,
                      int imageHeight,
                    float theta)
{
    /* Compute image center */
    float x0 = imageWidth * 0.5f;
    float y0 = imageHeight * 0.5f;

    /* Get global ID for output coordinates */
    /* i.e. exactly 1 work-item rotates a given pixel */
    for (int x = get_global_id(0); x < imageWidth; x += get_global_size(0)) {
        for (int y = get_global_id(1); y < imageHeight; y += get_global_size(1)) {
            /* Compute the work-item's location relative to the image center */
            int xprime = x - x0;
            int yprime = y - y0;

            /* Compute sine and cosine */
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);

            /* Compute the input location i.e. before rotation */
            float2 readCoord;
            readCoord.x = xprime * cosTheta - yprime * sinTheta + x0;
            readCoord.y = xprime * sinTheta + yprime * cosTheta + y0;

            /* Read the input image */
            float4 value = read_imagef(inputImage, sampler, readCoord);

            /* write to the globally unique pixel in the output image */
            /* i.e. copy pixel from original location to the rotated location */
            write_imagef(outputImage, (int2)(x, y), value);
        }
    }
}

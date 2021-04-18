

kernel void imageToInt(read_only image2d_t float_image, write_only image2d_t int_image) {
    int2 coords;
    coords.x = get_global_id(0);
    coords.y = get_global_id(1);

    float4 color = read_imagef(float_image, coords);
    color = pow(color, 1/2.2f);
    color.w = 1;
    write_imagef(int_image, coords, color);
}


kernel void calculatePixel(global uchar4 * picture, int width) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    picture[i * width + j].r = i/2;
    picture[i * width + j].g = j/2;
    picture[i * width + j].b =   0;
    picture[i * width + j].a = 255;
}
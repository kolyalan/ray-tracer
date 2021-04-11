#define R123_USE_U01_DOUBLE 0
#include "../kernels/Random123/u01fixedpt.h"
#include "../kernels/Random123/threefry.h"

typedef enum {
    INITIAL, INTERSECTED, ENDED
} ray_type;

typedef struct {
    float3 startPoint;
    float3 direction;
    ray_type type;
    int2 screenCoords;
    float weight;
} Ray;

float3 canvasToViewport(float3 viewVector, int2 coords, int screenWidth, int screenHeight, float distance) {
    float3 hor = viewVector;
    hor.y = 0;
    hor.x = viewVector.z;
    hor.z = viewVector.x;
    hor = normalize(hor);
    float3 vert = normalize(cross(viewVector, hor));
    
    float3 viewportCoords;
    viewportCoords.x = (float)coords.x/screenWidth-0.5f;
    viewportCoords.y = (float)coords.y/screenHeight-0.5f;
    viewportCoords.z = length(viewVector);
    viewportCoords = viewportCoords.x * hor + viewportCoords.y * vert + viewportCoords.z * normalize(viewVector);
    return viewportCoords;
}

float3 canvasToViewportOld(float3 viewVector, int2 coords, int screenWidth, int screenHeight, float distance) {
    float3 viewportCoords;
    viewportCoords.x = (float)coords.x/screenWidth-0.5f;
    viewportCoords.y = (float)coords.y/screenHeight-0.5f;
    viewportCoords.z = distance;
    return viewportCoords;
}

float3 randomShift(float3 vec, float3 normal, float radius, unsigned global_id, unsigned useed) {
    threefry4x32_key_t k = {{global_id, useed}};
    static threefry4x32_ctr_t c = {{}};
    union {
        threefry4x32_ctr_t c;
        int4 i;
    } u;
    c.v[0]++;
    u.c = threefry4x32(c, k);
    float3 shiftVec;
    shiftVec.x = u01fixedpt_closed_closed_32_float(u.i.x)*2-1;
    shiftVec.y = u01fixedpt_closed_closed_32_float(u.i.y)*2-1;
    shiftVec.z = u01fixedpt_closed_closed_32_float(u.i.z)*2-1;
    radius *= u01fixedpt_closed_closed_32_float(u.i.w);

    normal = normalize(normal);
    shiftVec = shiftVec - normal * dot(normal, shiftVec);
    shiftVec = normalize(shiftVec) * radius;
    return vec + shiftVec;
}

float random1D(unsigned global_id, unsigned useed) {
    threefry2x32_key_t k = {{global_id, useed}};
    static threefry2x32_ctr_t c = {{}};
    union {
        threefry2x32_ctr_t c;
        int2 i;
    } u;
    c.v[0]++;
    u.c = threefry2x32(c, k);
    return u01fixedpt_closed_closed_32_float(u.i.x);
}

float2 random2D(unsigned global_id, unsigned useed) {
    threefry2x32_key_t k = {{global_id, useed}};
    static threefry2x32_ctr_t c = {{}};
    union {
        threefry2x32_ctr_t c;
        int2 i;
    } u;
    c.v[0]++;
    u.c = threefry2x32(c, k);
    float2 ret = {u01fixedpt_closed_closed_32_float(u.i.x), u01fixedpt_closed_closed_32_float(u.i.y)};
    return ret;
}

float3 random3D(unsigned global_id, unsigned useed) {
    threefry4x32_key_t k = {{global_id, useed}};
    static threefry4x32_ctr_t c = {{}};
    union {
        threefry4x32_ctr_t c;
        int4 i;
    } u;
    c.v[0]++;
    u.c = threefry4x32(c, k);
    float3 ret = {u01fixedpt_closed_closed_32_float(u.i.x), u01fixedpt_closed_closed_32_float(u.i.y), u01fixedpt_closed_closed_32_float(u.i.z)};
    return ret;
}

kernel void emitRays(global Ray *rayList, int width, int height, int nRaysPerPixel,
                     float3 viewPoint, float3 viewVector, unsigned seed) {
    int2 coords;
    coords.x = get_global_id(0);
    coords.y = get_global_id(1);
    int global_id = coords.y * get_global_size(0) + coords.x;
    float3 viewportCoords = canvasToViewport(viewVector, coords, width, height, 1);
    for (int i = 0; i < nRaysPerPixel; i++) {
        float3 shiftedviewportCoords = randomShift(viewportCoords, viewVector, 0.001f, global_id, seed);
        float3 shiftedViewPoint = randomShift(viewPoint, viewVector, 0.02f, global_id, seed);
        Ray curRay = {shiftedViewPoint, shiftedviewportCoords - shiftedViewPoint, INITIAL, coords, 1.0f/nRaysPerPixel};
        rayList[i*width*height + coords.x * width + coords.y] = curRay;
    }

/*
    float4 color = {coords.x/(float)width, 0, coords.y/(float)width, 1};
    write_imagef(image, coords, color);
*/
}
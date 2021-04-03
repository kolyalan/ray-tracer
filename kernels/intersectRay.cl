
typedef struct {
    float3 center;
    float radius;
    float4 color;
    int specular;
    float reflective;
} Sphere;

typedef struct {
    float3 Position;
    float3 Normal;
    float2 TexCoords;
} Vertex;

typedef struct {
    uint vertex0;
    uint vertex1;
    uint vertex2;
} Triangle;

typedef enum  {
    AMBIENT, POINT, DIRECTIONAL
} LightType;

typedef struct {
    LightType type;
    float intensity;
    float3 dir;
} Light;

typedef struct {
    int vertices;
    int indices;
    int triangleNum;
    int texture_offset;
    int textureIdx;
    float4 specular;
} Mesh_idx;


typedef struct {
    Mesh_idx * meshes;
    int meshesNum;
    Vertex *vertices;
    Triangle *triangles;
    Light *lights;
    int lightsNum;
} Scene;

const sampler_t mySampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

void intersectRaySphere(float3 Ov, float3 D, Sphere ball, float * t1, float *t2) {
    float3 C = ball.center;
    float r = ball.radius;
    float3 OC = Ov - C ;//возможно, поменять местами

    float k1 = dot(D, D);
    float k2 = 2*dot(OC, D);
    float k3 = dot(OC, OC) - r*r;

    float discr = k2*k2 - 4*k1*k3;
    if (discr < 0) {
        *t1 = INFINITY;
        *t2 = INFINITY;
    } else {
        *t1 = (- k2 - sqrt(discr))/(2*k1);
        *t2 = (- k2 + sqrt(discr))/(2*k1);
    }
}
/*
bool intersectRayTriangle(float3 rayOrigin, 
                          float3 rayVector, 
                          Triangle* inTriangle,
                          float3 *outIntersectionPoint)
{
    const float EPSILON = 0.0000001;
    float3 vertex0 = inTriangle->vertex0;
    float3 vertex1 = inTriangle->vertex1;  
    float3 vertex2 = inTriangle->vertex2;
    float3 edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = cross(rayVector, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.
    f = 1.0 / a;
    s = rayOrigin - vertex0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = cross(s, edge1);
    v = f * dot(rayVector, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > EPSILON) // ray intersection
    {
        *outIntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}*/

#define kEpsilon 0.0000001f

bool rayTriangleIntersect( 
    const float3 orig, const float3 dir, bool frontOnly,
    const float3 v0, const float3 v1, const float3 v2, 
    float *t, float *u, float *v) { 
    float3 v0v1 = v1 - v0; 
    float3 v0v2 = v2 - v0; 
    float3 pvec = cross(dir, v0v2); 
    float det = dot(v0v1, pvec); 

    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (frontOnly && det < kEpsilon) return false; 

    // ray and triangle are parallel if det is close to 0
    if (!frontOnly && fabs(det) < kEpsilon) return false; 

    float invDet = 1 / det; 
 
    float3 tvec = orig - v0; 
    *u = dot(tvec, pvec) * invDet; 
    if (*u < 0 || *u > 1) return false; 
 
    float3 qvec = cross(tvec, v0v1); 
    *v = dot(dir, qvec) * invDet; 
    if (*v < 0 || *u + *v > 1) return false; 
 
    *t = dot(v0v2, qvec) * invDet; 
 
    return true;  
}

bool closestIntersection (float3 orig, float3 dir, bool frontOnly, float t_min, float t_max, Scene scene, int *objectId, float *t, float4 *texCoords, float3 *normal) {
    float t_closest = INFINITY;
    Vertex vertex0, vertex1, vertex2;
    int mesh_min = -1;
    float u, v;
    for (int j = 0; j < scene.meshesNum; j++) {
        Mesh_idx mesh = scene.meshes[j];
        int tOff = mesh.indices;
        int vOff = mesh.vertices;
        for (int i = 0; i < mesh.triangleNum; i++) {
            Vertex tmpVertex0 = scene.vertices[vOff+scene.triangles[tOff+i].vertex0];
            Vertex tmpVertex1 = scene.vertices[vOff+scene.triangles[tOff+i].vertex1];
            Vertex tmpVertex2 = scene.vertices[vOff+scene.triangles[tOff+i].vertex2];
            
            float t, tmpU, tmpV;
            if (rayTriangleIntersect(orig, dir, frontOnly,
                                    tmpVertex0.Position,
                                    tmpVertex1.Position,
                                    tmpVertex2.Position,
                                    &t, &tmpU, &tmpV)) {
                if (t > t_min && t < t_max && t_closest > t) {
                    t_closest = t;
                    vertex0 = tmpVertex0;
                    vertex1 = tmpVertex1;
                    vertex2 = tmpVertex2;
                    u = tmpU;
                    v = tmpV;
                    mesh_min = j;
                }
            }
        }
    }
    if (objectId != NULL) {
        *objectId = mesh_min;
    }
    if (t != NULL) {
        *t = t_closest;
    }
    if (texCoords != NULL) {
        texCoords->xy = vertex1.TexCoords * u + vertex2.TexCoords * v + vertex0.TexCoords * (1-u-v);
        texCoords->z = scene.meshes[mesh_min].textureIdx;
    }
    if (normal != NULL) {
        *normal = vertex1.Normal * u + vertex2.Normal * v + vertex0.Normal * (1-u-v);
    }
    return (mesh_min != -1);
}

bool findIntersection(float3 orig, float3 dir, bool frontOnly, float t_min, float t_max, Scene scene, image2d_array_t texture,
                      float4 *diffuseColor, float3 *coords, float3 *normal, float *specular) { // Add out parameters if needed
    int objectId;
    float t;
    float u, v;
    Vertex vertex0, vertex1, vertex2;
    float4 texCoords;

    if (closestIntersection(orig, dir, frontOnly, t_min, t_max, scene, &objectId, &t, &texCoords, normal)) {
        float4 clr = {u, v, 1-u-v, 1};
        if (diffuseColor != NULL) {
            *diffuseColor = read_imagef(texture, mySampler, texCoords);
        }
        if (coords != NULL) {
            *coords = orig + dir * t;
        }
        if (specular != NULL) {
            *specular = length(scene.meshes[objectId].specular)/3;
        }
        return true;
    } else {
        float4 background = {0.2f, 0.2f, 0.8f, 1.0f};
        if (diffuseColor != NULL) {
            *diffuseColor = background;
        }
        return false;
    }
}

float4 computeLightning(float3 orig, float3 dir, Scene scene, float4 diffuseColor, float3 coords, float3 normal, float specular, image2d_array_t texture) {
    float3 viewVector = dir;
    float4 color = 0;
    float3 lightVector=0;
    float distance = 1;
    float t_max;
    coords += normal * 0.00001f;
    for (int i = 0; i < scene.lightsNum; i++) {
        if(scene.lights[i].type == AMBIENT) {
                color += diffuseColor * scene.lights[i].intensity;
        } else {    
            if (scene.lights[i].type == POINT) {
                lightVector = scene.lights[i].dir - coords;
                distance = length(scene.lights[i].dir);
                distance *= distance;
                distance /= 5;
                t_max = 1;
            } else {// DIRECTIONAL
                lightVector = scene.lights[i].dir;
                t_max = INFINITY;
            }
            float shadow = 1;
            
            float t;
            int objectId;
            float4 texCoords;
            if (closestIntersection(coords, lightVector, false, 0, t_max, scene, &objectId, &t, &texCoords, NULL)) {
                float4 diffuseColor = read_imagef(texture, mySampler, texCoords);
                shadow = 1 - diffuseColor.a;
            }

            float nDotL = dot(normal, lightVector);
            if (nDotL > 0) {
                color += diffuseColor * (scene.lights[i].intensity * shadow* nDotL / (length(normal) * length(lightVector)) / distance);
            }
            
            if (specular > kEpsilon) {
                float3 R = 2*normal*dot(normal, -lightVector) + lightVector;
                float RDotV = dot(R, viewVector);
                if (RDotV > 0) {
                    color += diffuseColor * (scene.lights[i].intensity * shadow * 5 * specular * pow(RDotV/(length(R)*length(viewVector)), 350*specular) / distance) ;
                }
            }
        }
    }
    color.a = 1;
    return color;
}

kernel void intersectRay(global Ray *rayList, read_write image2d_t screen, global Mesh_idx * meshes, int meshesNum,
                         global Vertex *vertices, global Triangle *triangles, read_only image2d_array_t texture,
                         global Light *lights, int lightsNum) {
    int globalId = get_global_id(0);
    Ray myRay = rayList[globalId];

    float4 color = {0, 0, 0, 1};
    float4 diffuseColor;
    float3 coords;
    float3 normal;
    float specular;
    Scene scene = {meshes, meshesNum, vertices, triangles, lights, lightsNum};

    if (!findIntersection(myRay.startPoint, myRay.direction, true, 0, INFINITY, scene, texture,
                          &diffuseColor, &coords, &normal, &specular)) {
        color.xyz = normalize(myRay.direction)/2 + 0.5f;
    } else {
        color = computeLightning(myRay.startPoint, myRay.direction, scene, diffuseColor, coords, normal, specular, texture);
    }

    float weight = myRay.weight;

    float4 oldColor = read_imagef(screen, myRay.screenCoords);
    color = color*weight + oldColor;
    write_imagef(screen, myRay.screenCoords, color);
}
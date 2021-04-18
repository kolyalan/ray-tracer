

#define N_IN 1.6f
#define N_OUT 1.0f

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
    float *t, float *u, float *v, bool *frontFacing) { 
    float3 v0v1 = v1 - v0; 
    float3 v0v2 = v2 - v0; 
    float3 pvec = cross(dir, v0v2); 
    float det = dot(v0v1, pvec); 

    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    *frontFacing = !(det < kEpsilon);
    if (!*frontFacing && frontOnly) {
        return false; 
    }

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
    bool frontFacing = 0;
    for (int j = 0; j < scene.meshesNum; j++) {
        Mesh_idx mesh = scene.meshes[j];
        int tOff = mesh.indices;
        int vOff = mesh.vertices;
        for (int i = 0; i < mesh.triangleNum; i++) {
            Vertex tmpVertex0 = scene.vertices[vOff+scene.triangles[tOff+i].vertex0];
            Vertex tmpVertex1 = scene.vertices[vOff+scene.triangles[tOff+i].vertex1];
            Vertex tmpVertex2 = scene.vertices[vOff+scene.triangles[tOff+i].vertex2];
            
            float t, tmpU, tmpV;
            bool tmpFrontFacing = 0;
            if (rayTriangleIntersect(orig, dir, frontOnly,
                                    tmpVertex0.Position,
                                    tmpVertex1.Position,
                                    tmpVertex2.Position,
                                    &t, &tmpU, &tmpV, &tmpFrontFacing)) {
                if (t > t_min && t < t_max && t_closest > t) {
                    t_closest = t;
                    vertex0 = tmpVertex0;
                    vertex1 = tmpVertex1;
                    vertex2 = tmpVertex2;
                    u = tmpU;
                    v = tmpV;
                    mesh_min = j;
                    frontFacing = tmpFrontFacing;
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
    float lightIntensity = 0;
    float3 lightVector=0;
    float distance = 1;
    float t_max;
    coords += normal * 0.00001f;
    for (int i = 0; i < scene.lightsNum; i++) {
        if(scene.lights[i].type == AMBIENT) {
                lightIntensity += scene.lights[i].intensity;
        } else {    
            if (scene.lights[i].type == POINT) {
                lightVector = scene.lights[i].dir - coords;
                distance = length(scene.lights[i].dir);
                distance *= distance;
                //distance /= 5;
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
                //float4 diffuseColor = read_imagef(texture, mySampler, texCoords);
                //shadow = 1 - diffuseColor.a;
                shadow = 0;
            }

            float nDotL = dot(normal, lightVector);
            if (nDotL > 0) {
                lightIntensity += scene.lights[i].intensity * shadow* nDotL / (length(normal) * length(lightVector)) / distance;
            }
            
            if (specular > kEpsilon) {
                float3 R = 2*normal*dot(normal, -lightVector) + lightVector;
                float RDotV = dot(R, viewVector);
                if (RDotV > 0) {
                    lightIntensity += scene.lights[i].intensity * shadow * 50 * specular * pow(RDotV/(length(R)*length(viewVector)), 100*specular) / distance;
                }
            }
        }
    }
    color = diffuseColor * lightIntensity;
    color.a = 1;
    return color;
}

float3 refract (float3 I, float3 N, float eta) {
    //For a given incident vector I, surface normal N and ratio of 
    //indices of refraction, eta, refract returns the refraction vector, R. 
    //R is calculated as:
    float3 R;
    float k = 1.0f - eta * eta * (1.0f - dot(N, I) * dot(N, I));
    if (k < 0.0f)
        R = 0.0f;
    else
        R = eta * I - (eta * dot(N, I) + sqrt(k)) * N;

    //The input parameters I and N should be normalized in order to 
    //achieve the desired result.
    return R;
}

float3 reflect(float3 I, float3 N) {
    return I - 2.0f * dot(N, I) * N;
}

float3 RandomHemispherePoint(float2 rand) {
    float cosTheta = sqrt(1.0f - rand.x);
    float sinTheta = sqrt(rand.x);
    float phi = 2.0f * M_PI_F * rand.y;
    float3 R = {cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta};
    return R;
}

float3 NormalOrientedHemispherePoint(float2 rand, float3 n) {
    float3 v = RandomHemispherePoint(rand);
    return dot(v, n) < 0.0f ? -v : v;
}

float FresnelSchlick(float nIn, float nOut, float angle) {
    float R0 = ((nOut - nIn) * (nOut - nIn)) / ((nOut + nIn) * (nOut + nIn));
    float fresnel = R0 + (1.0f - R0) * pow((1.0f - cos(angle)), 5.0f);
    return fresnel;
}

float3 IdealRefract(float3 direction, float3 normal, float nIn, float nOut) {
    // проверим, находимся ли мы внутри объекта
    // если да - учтем это при расчете сред и направления луча
    bool fromOutside = dot(normal, direction) < 0.0f;
    float ratio = fromOutside ? nOut / nIn : nIn / nOut;

    float3 refraction, reflection;
    refraction = fromOutside ? refract(direction, normal, ratio) : -refract(-direction, normal, ratio);
    reflection = reflect(direction, normal);

    // в случае полного внутренного отражения refract вернет нам 0.0
    return refraction == 0.0f ? reflection : refraction;
}

bool IsRefracted(float rand, float3 direction, float3 normal, float opacity, float nIn, float nOut) {
    bool fromOutside = dot(normal, direction) < 0.0f;
    float ratio = fromOutside ? nOut / nIn : nIn / nOut;
    float angle = acos(fabs(dot(direction, normal)));
    if (ratio > 1) {
        float critAngle = asin(1/ratio);
        if (angle > critAngle) {
            return false;
        }
        angle = angle / critAngle * M_PI_2_F;
    }
    float fresnel = FresnelSchlick(nIn, nOut, angle);
    return opacity > rand && fresnel < rand;
}

#define ITERATIONS 10
#define SPEED 10.0f
#define DISPLACEMENT 0.06f
#define TIGHTNESS 15.0f
#define YOFFSET 0.1f
#define YSCALE 0.25f
#define FLAMETONE {50.0f, 5.0f, 1.0f}

float shape(float2 pos) // a blob shape to distort
{
	return clamp( sin(pos.x*3.14169265f) - pos.y+YOFFSET, 0.0f, 1.0f );
}

float noise(float3 x, image2d_array_t texture) // iq noise function
{
	float3 p;
    float3 f = fract(x, &p);
	f = f*f*(3.0f-2.0f*f);
    float2 idk = {37.0f,17.0f};
	float2 uv = (p.xy+idk*p.z) + f.xy;
    float4 texCoords = {(uv+ 0.5f)/256.0f, 4, 0};
	float2 rg = read_imagef(texture, mySampler, texCoords).yx;
	return mix( rg.x, rg.y, f.z ) * 2.0f - 1.0f;
}

float plaIntersect(float3 ro, float3 rd, float4 p )
{
    return -(dot(ro,p.xyz)+p.w)/dot(rd,p.xyz);
}

float4 marchRay(float3 startPoint, float3 endPoint, image2d_array_t texture) {
    float3 dir = normalize(endPoint - startPoint);
    float4 p = {0, 0, 1, 0};
    float2 uv = (startPoint + dir*plaIntersect(startPoint, dir, p)).xy;
    //uv.y += 0.5;
    uv = (uv+0.5f)/0.62f;
    uv.y = 1-uv.y+0.5;
    //float4 clr = {uv, 0.0f, 1.0f};
    //return clr;
	float nx = 0.0f;
	float ny = 0.0f;
    float iTime = 1;
	for (int i=1; i<ITERATIONS+1; i++)
	{
		float ii = pow((float)i, 2.0f);
		float ifrac = (float)i/(float)ITERATIONS;
		float t = ifrac * iTime * SPEED;
		float d = (1.0f-ifrac) * DISPLACEMENT;
        float3 t1 = {uv.x*ii-iTime*ifrac, uv.y*YSCALE*ii-t, 0.0f};
        float3 t2 = {uv.x*ii+iTime*ifrac, uv.y*YSCALE*ii-t, iTime*ifrac/ii};
		nx += noise(t1, texture) * d * 2.0f;
		ny += noise(t2, texture) * d;
	}
    float2 uvnxy = {uv.x+nx, uv.y+ny};
	float flame = shape(uvnxy);
    float3 tone = FLAMETONE;
	float3 col = pow(flame, TIGHTNESS) * tone;
    
    // tonemapping
    col = col / (1.0f+col);
    col = pow(col, 1.0f/2.2f);
    col = clamp(col, 0.0f, 1.0f);
    float alpha = 2*pow(length(startPoint - endPoint)/0.62f, 15);
	
    float4 color = {10*col, alpha};
	return color;
/*
    float4 color = 0;
    float3 dir = normalize(endPoint - startPoint)*0.1f;
    float3 point = startPoint+dir;
    int n = length(endPoint - point) / length(dir);
    for (int i = 0; i < n; i++, point += dir) {
        float intensity = (NoiseInPoint(point/10.0f)+1)/2;
        float4 local_color = {intensity, 0, 0, 1};
        color += local_color;
    }
    //color /= n/100.0f;
    color *= pow(length(startPoint - endPoint)/0.62f, 10);
    color.a = 1.0f;
    return color;*/
}

kernel void intersectRay(global Ray *rayList, read_write image2d_t screen, global Mesh_idx * meshes, int meshesNum,
                         global Vertex *vertices, global Triangle *triangles, read_only image2d_array_t texture,
                         global Light *lights, int lightsNum, unsigned useed, int iteration) {
    int globalId = get_global_id(0);
    Ray myRay = rayList[globalId];

    if (myRay.type == ENDED) {
        return;
    }

    float4 color = {0, 0, 0, 1};
    float4 diffuseColor;
    float3 coords;
    float3 normal;
    float specular;
    Scene scene = {meshes, meshesNum, vertices, triangles, lights, lightsNum};

    float weight = myRay.weight;

    if (!findIntersection(myRay.startPoint, myRay.direction, false, 0, INFINITY, scene, texture,
                          &diffuseColor, &coords, &normal, &specular)) {
        //color.xyz = {0, 0, 0}; //normalize(myRay.direction)/2 + 0.5f;
        myRay.type = ENDED;
        color *= myRay.weight;
        rayList[globalId] = myRay;
    } else {
        color = computeLightning(myRay.startPoint, myRay.direction, scene, diffuseColor, coords, normal, specular, texture);
        float fade = 0;
        if (myRay.type == MARCH) {
            float4 fire_color = marchRay(myRay.startPoint, coords, texture);
            color.xyz = mix(color.xyz, fire_color.xyz, fire_color.w);
            fade = fire_color.w;

        }

        if (specular > 0.0f) {
            float3 rayDirection = myRay.direction;
            float3 newRayOrigin = coords;
            float3 hemisphereDistributedDirection = NormalOrientedHemispherePoint(random2D(globalId, useed), normal);

            float3 randomVec = normalize(2.0f * random3D(globalId, useed) - 1.0f);
            float3 tangent = cross(randomVec, normal);
            float3 bitangent = cross(normal, tangent);
            float3 transform[3] = {tangent, bitangent, normal};
            float3 transform2[3] = {{transform[0][0], transform[1][0], transform[2][0]},
                                    {transform[0][1], transform[1][1], transform[2][1]},
                                    {transform[0][2], transform[1][2], transform[2][2]}};
            float3 newRayDirection = {dot(transform2[0], hemisphereDistributedDirection),
                                      dot(transform2[1], hemisphereDistributedDirection),
                                      dot(transform2[2], hemisphereDistributedDirection)};
            newRayDirection = normalize(newRayDirection);

            float roughness = 1;
            bool refracted = IsRefracted(random1D(globalId, useed), normalize(rayDirection), normalize(normal), (1.0f - diffuseColor.a), N_IN, N_OUT);
            if (refracted)
            {
                float3 idealRefraction = IdealRefract(normalize(rayDirection), normal, N_IN, N_OUT);
                newRayDirection = normalize(mix(-newRayDirection, idealRefraction, roughness));
                newRayOrigin += normal * (dot(newRayDirection, normal) < 0.0f ? -0.00001f : 0.00001f);
            }
            else
            {
                float3 idealReflection = reflect(normalize(rayDirection), normal);
                newRayDirection = normalize(mix(newRayDirection, idealReflection, roughness));
                newRayOrigin += normal * 0.000001f;
            }
            myRay.type = (dot(newRayDirection, normal) < 0.0f) ? MARCH : INITIAL;
            myRay.direction = newRayDirection;
            myRay.startPoint = newRayOrigin;

            color *= myRay.weight*(1-specular);
            myRay.weight *= specular;
            if (myRay.type == MARCH) {
                myRay.weight*=(1-fade);
            }
            rayList[globalId] = myRay;
        } else {
            myRay.type = ENDED;
            color *= myRay.weight;
            rayList[globalId] = myRay;
        }
    }

    float4 oldColor = read_imagef(screen, myRay.screenCoords);
    color += oldColor;
    write_imagef(screen, myRay.screenCoords, color);
}
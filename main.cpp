#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#define CL_VERSION_2_2
#include "CL/cl2.hpp"
#include "Image.h"
#include "mesh.h"

extern char *optarg;
extern int optind, opterr, optopt;

int ScreenWidth = 512, ScreenHeight = 512;

static cl::Platform defaultPlatform;
static cl::Device defaultDevice;
cl::Context context;
cl::CommandQueue queue;

typedef struct {
    cl_float3 center;
    float radius;
    cl_float4 color;
    int specular;
    float reflective;
} Sphere;

typedef enum {
     ENDED, INITIAL
} ray_type;

typedef struct {
    cl_float3 startPoint;
    cl_float3 direction;
    ray_type type;
    cl_int2 screenCoords;
    float weight;
} Ray;

typedef enum  {
    AMBIENT, POINT, DIRECTIONAL
} LightType;

typedef struct {
    LightType type;
    float intensity;
    cl_float3 dir;
} Light;

int initOpenCL() {
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        return 1;
    }
    std::cout << "Found " << allPlatforms.size() << " platform(s)" << std::endl;
    defaultPlatform = allPlatforms[0];
    std::cout << "Using platform " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    //get default device of the default platform
    std::vector<cl::Device> allDevices;
    defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
    if (allDevices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        return 1;
    }
    std::cout << "Found " << allDevices.size() << " device(s)" << std::endl;
    defaultDevice = allDevices[0];
    std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
    //int workGroupMultiple = defaultDevice.getInfo<CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>();
    int maxWorkGroupSize = defaultDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::cout << defaultDevice.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;

    context = cl::Context(defaultDevice);

    //create queue to which we will push commands for the device.
    queue =  cl::CommandQueue(context, defaultDevice);
    
    return 0;
}

cl::Program buildCLProgram(cl_int *err) {
    cl::Program::Sources sources;

    std::ifstream ifs("../kernels/emitRays.cl");
    std::string emitRaysCode((std::istreambuf_iterator<char>(ifs)),
                           (std::istreambuf_iterator<char>()));
    sources.push_back(emitRaysCode);
    ifs.close();

    ifs.open("../kernels/intersectRay.cl");
    std::string intersectRaysCode((std::istreambuf_iterator<char>(ifs)),
                           (std::istreambuf_iterator<char>()));
    sources.push_back(intersectRaysCode);
    ifs.close();

    ifs.open("../kernels/sortRays.cl");
    std::string sortRaysCode((std::istreambuf_iterator<char>(ifs)),
                           (std::istreambuf_iterator<char>()));
    sources.push_back(sortRaysCode);
    ifs.close();

    ifs.open("../kernels/castRays.cl");
    std::string castRaysCode((std::istreambuf_iterator<char>(ifs)),
                           (std::istreambuf_iterator<char>()));
    sources.push_back(castRaysCode);
    ifs.close();

        ifs.open("../kernels/imageToInt.cl");
    std::string imageToIntCode((std::istreambuf_iterator<char>(ifs)),
                           (std::istreambuf_iterator<char>()));
    sources.push_back(imageToIntCode);
    ifs.close();

    cl::Program program(context, sources);
    if(program.build({defaultDevice}, "-cl-std=CL2.0 ") != CL_SUCCESS){
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << "\n";
        *err = 1;
    }
    *err = 0;
    return program;
}

size_t getGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
	size_t r = DataElemCount % LocalWorkSize;
	if(r == 0)
		return DataElemCount;
	else
		return DataElemCount + LocalWorkSize - r;
}

int callCLSort(cl::Program &program, size_t N_padded, size_t LocalWorkSize,
                cl::Buffer &m_dPingArray, cl::Buffer &m_dPongArray, int shift) {
    cl_int clError, err_no = 0;
	size_t globalWorkSize;
	size_t localWorkSize;

	localWorkSize = LocalWorkSize;
	globalWorkSize = getGlobalWorkSize(N_padded / 2, localWorkSize);
	unsigned int limit = (unsigned int)2 * LocalWorkSize; //limit is double the localWorkSize

	// start with Sort_BitonicMergesortLocalBegin to sort local until we reach the limit
    static cl::Kernel m_BitonicStartKernel(program, "Sort_BitonicMergesortStart", &err_no);
    if (err_no != CL_SUCCESS) {
        std::cout << " Error creating kernel Sort_BitonicMergesortStart: " << err_no << std::endl;
        return 1;
    }
    static cl::Kernel m_BitonicGlobalKernel(program, "Sort_BitonicMergesortGlobal", &err_no);
    if (err_no != CL_SUCCESS) {
        std::cout << " Error creating kernel Sort_BitonicMergesortGlobal: " << err_no << std::endl;
        return 1;
    }
    static cl::Kernel m_BitonicLocalKernel(program, "Sort_BitonicMergesortLocal", &err_no);
    if (err_no != CL_SUCCESS) {
        std::cout << " Error creating kernel Sort_BitonicMergesortLocal: " << err_no << std::endl;
        return 1;
    }
	m_BitonicStartKernel.setArg(0, m_dPingArray);
	m_BitonicStartKernel.setArg(1, m_dPongArray);
	m_BitonicStartKernel.setArg(2, shift);
    
    queue.enqueueNDRangeKernel(m_BitonicStartKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));

	// proceed with global and local kernels
	for (unsigned int blocksize = limit; blocksize <= N_padded; blocksize <<= 1) {
		for (unsigned int stride = blocksize / 2; stride > 0; stride >>= 1) {
			if (stride >= limit) {
				//Sort_BitonicMergesortGlobal
				m_BitonicGlobalKernel.setArg(0, m_dPongArray);
				m_BitonicGlobalKernel.setArg(1, N_padded);
				m_BitonicGlobalKernel.setArg(2, blocksize);
				m_BitonicGlobalKernel.setArg(3, stride);
				m_BitonicGlobalKernel.setArg(4, shift);
                
                queue.enqueueNDRangeKernel(m_BitonicGlobalKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));
			} else {
				//Sort_BitonicMergesortLocal
				m_BitonicLocalKernel.setArg(0,m_dPongArray);
				m_BitonicLocalKernel.setArg(1,N_padded);
				m_BitonicLocalKernel.setArg(2,blocksize);
				m_BitonicLocalKernel.setArg(3,stride);
				m_BitonicLocalKernel.setArg(4,shift);
                
                
                queue.enqueueNDRangeKernel(m_BitonicLocalKernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize));
			}
		}
	}
	//std::swap(m_dPingArray, m_dPongArray);
    return 0;
}

int callCLTraceRay(MyImage &screen, std::vector<Light> &lights, cl_float3 viewPoint, cl_float3 viewVector, Model &model,
                    int depth, int nRaysPerPixel) {
    
    cl_int err_no;
    cl::Program program = buildCLProgram(&err_no);
    if (err_no != 0) {
        return 1;
    }
    int nPixels = ScreenWidth*ScreenHeight;

	unsigned int log2val = (unsigned int)ceil(log((float)nPixels*nRaysPerPixel) / log(2.f));
	size_t N_padded =  (size_t)pow(2, log2val);
    //std::cout << nPixels*nRaysPerPixel << " -> " << N_padded << std::endl;

    // create buffers on the device
    //cl::Buffer screenBuffer(context, CL_MEM_READ_WRITE, screen.getSize());
    cl::Image2D floatScreenBuffer(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), ScreenWidth, ScreenHeight);
    cl::Image2D int8ScreenBuffer(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), ScreenWidth, ScreenHeight);
    
    cl::Buffer vertexesBuffer(context, model.vertices.begin(), model.vertices.end(), true, false, &err_no);
    if (err_no != 0) {
        std::cout << "Unable to create vertex buffer. err: " << err_no << std::endl;
        return 1;
    }

    cl::Buffer indicesBuffer(context, model.indices.begin(), model.indices.end(), true, false, &err_no);
    if (err_no != 0) {
        std::cout << "Unable to create indices buffer. err: " << err_no << std::endl;
        return 1;
    }

    cl::Buffer meshesBuffer(context, model.mesh_index.begin(), model.mesh_index.end(), &err_no);
    if (err_no != 0) {
        std::cout << "Unable to create meshes buffer. err: " << err_no << std::endl;
        return 1;
    }

    cl::Buffer lightsBuffer(context, lights.begin(), lights.end(), true, false, &err_no);
    if (err_no != 0) {
        std::cout << "Unable to create ligths buffer. err: " << err_no << std::endl;
        return 1;
    }

    cl::Image2DArray *texture = model.textures;

    cl::Buffer rayPool(context, CL_MEM_READ_WRITE, N_padded*sizeof(Ray));
    cl::Buffer tmpRayPool(context, CL_MEM_READ_WRITE, N_padded*sizeof(Ray));

    //write arrays A and B to the device
    //queue.enqueueWriteImage(screenBuffer, CL_TRUE, screen.getData(), );

    cl::Kernel emit_kernel(program, "emitRays", &err_no);
    if (err_no != CL_SUCCESS) {
        std::cout << " Error creating kernel emitRays: " << err_no << std::endl;
        return 1;
    }

    emit_kernel.setArg(0, rayPool);
    emit_kernel.setArg(1, ScreenWidth);
    emit_kernel.setArg(2, ScreenHeight);
    emit_kernel.setArg(3, nRaysPerPixel);
    emit_kernel.setArg(4, viewPoint);
    emit_kernel.setArg(5, viewVector);
    emit_kernel.setArg(6, (unsigned)rand());

    queue.enqueueNDRangeKernel(emit_kernel, cl::NullRange, cl::NDRange(ScreenWidth, ScreenHeight), cl::NullRange);


    cl::Kernel intersectKernel(program, "intersectRay", &err_no);
    if (err_no != CL_SUCCESS) {
        std::cout << " Error creating kernel emitRays: " << err_no << std::endl;
        return 1;
    }

    intersectKernel.setArg(1, floatScreenBuffer);
    intersectKernel.setArg(2, meshesBuffer);
    intersectKernel.setArg(3, model.mesh_index.size());
    intersectKernel.setArg(4, vertexesBuffer);
    intersectKernel.setArg(5, indicesBuffer);
    intersectKernel.setArg(6, *texture);
    intersectKernel.setArg(7, lightsBuffer);
    intersectKernel.setArg(8, lights.size());

    for (int iteration = depth; iteration > 0; iteration--) {
        intersectKernel.setArg(0, rayPool);
        intersectKernel.setArg(9, (unsigned)rand());
        intersectKernel.setArg(10, iteration);
        for (int i = 0; i < nRaysPerPixel; i++) {
            err_no = queue.enqueueNDRangeKernel(intersectKernel, cl::NDRange(i*nPixels), cl::NDRange(nPixels), cl::NullRange);
            if (err_no != CL_SUCCESS) {
                std::cout << "Unable to call intersectRays Err: " << err_no << std::endl;
                return 1;
            }
        }
        #ifdef SORT_RAYS
        for (int i = 0; i < nRaysPerPixel; i++) {
            callCLSort(program, nPixels, 256, rayPool, tmpRayPool, i*nPixels);
        }
        std::swap(rayPool, tmpRayPool);
        #endif

        /*Ray * rayList = new Ray[nPixels*nRaysPerPixel];

        queue.enqueueReadBuffer(rayPool, 1, 0, sizeof(*rayList)*nPixels*nRaysPerPixel, rayList);
        int ended = 0, initital = 0;
        for (int i = 0; i < nPixels*nRaysPerPixel; i++) {
                std::cout << rayList[i].type; // "x" << rayList[i].screenCoords.x << " y" << rayList[i].screenCoords.y << " " << rayList[i].type << std::endl;
                if (rayList[i].type == 0) {
                    ended++;
                }
                if (rayList[i].type == 1) {
                    initital++;
                }
        }
        std::cout << std::endl << "iteration: " << iteration << "; intitial rays: " << initital << "; ended rays: " << ended << std::endl; 
        delete rayList;*/
    }

    cl::Kernel imageToIntKerel(program, "imageToInt", &err_no);
    if (err_no != CL_SUCCESS) {
        std::cout << " Error creating kernel emitRays: " << err_no << std::endl;
        return 1;
    }

    imageToIntKerel.setArg(0, floatScreenBuffer);
    imageToIntKerel.setArg(1, int8ScreenBuffer);
    queue.enqueueNDRangeKernel(imageToIntKerel, cl::NullRange, cl::NDRange(ScreenWidth, ScreenHeight), cl::NullRange);

    //read result C from the device to array C
    queue.enqueueReadImage(int8ScreenBuffer, CL_TRUE, {0,0,0}, {(unsigned)ScreenWidth, (unsigned)ScreenHeight, 1}, 0, 0, screen.getData());

    return 0;
}
/*
cl_float3 canvasToViewport(cl_int2 coords, int screenWidth, int screenHeight, float distance = 1) {
    return {(float)coords.x/screenWidth/2, (float)coords.y/screenHeight/2, distance};
}
*/
int main(int argc, char *argv[]) {
    srand(time(NULL));
    initOpenCL();
    
    int raysPerPixel = 16;
    int depth = 4;
    int opt = 0;
    if (argc > 1) {
        while((opt = getopt(argc, argv, "w:s:d:")) != -1) {
            switch(opt) {
                case 'w':
                    ScreenHeight = ScreenWidth = atoi(optarg);
                    break;
                case 's':
                    raysPerPixel = atoi(optarg);
                    break;
                case 'd':
                    depth = atoi(optarg);
                    break;
                default:
                    break;
            }
        }
    }


    MyImage picture(ScreenWidth, ScreenHeight);

    Model model("../model/crystal_ball.obj");

    cl_float3 viewPoint = {0, -1.5, -10};
    cl_float3 viewVector = {0, 0.15/10, 1.0/10};
    std::vector<Light> lights;
    lights.push_back({AMBIENT, 0.3f, {0,0,0}});
    lights.push_back({POINT, 0.7f, {-1,-2,-1}});
   //lights.push_back({DIRECTIONAL, 0.15f, {0,-2,1}});
    lights.push_back({POINT, 0.7, {2, -1, -1}});

    if (callCLTraceRay(picture, lights, viewPoint, viewVector, model, depth, raysPerPixel)) {
        std::cout << "Some error" << std::endl;
    } else {
        std::cout << "Rendered successfully" << std::endl;
        picture.save("../327_lanbin_v0v0.png");
    }

    return 0;
}

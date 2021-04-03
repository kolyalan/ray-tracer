#ifndef GRAPHICS0_MESH_H
#define GRAPHICS0_MESH_H

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_VERSION_3_0
#include "CL/opencl.hpp"
#include <vector>
#include "Image.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

typedef struct {
    cl_float3 Position;
    cl_float3 Normal;
    cl_float2 TexCoords;
} Vertex;


struct Texture {
    MyImage texture;
    bool hasTexture;
    std::string path;

    Texture() {};
    Texture(const MyImage &tex, const std::string path) : texture(tex), hasTexture(true), path(path) {};
    Texture(const Texture &other) {
        texture = other.texture;
        hasTexture = other.hasTexture;
        path = other.path;
    }
    Texture(Texture &&other) {
        texture = std::move(other.texture);
        hasTexture = other.hasTexture;
        path = other.path;
    }

    Texture &operator=(const Texture &other) {
        if (this == &other) {
            return *this;
        }
        texture = other.texture;
        hasTexture = other.hasTexture;
        path = other.path;
        return *this;
    }

    Texture &operator=(Texture &&other) {
        if (this == &other) {
            return *this;
        }
        texture = std::move(other.texture);
        hasTexture = other.hasTexture;
        path = other.path;
        return *this;
    }

    ~Texture() {
    }
}; 

struct Mesh {
    /*  Mesh Data  */
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Texture texture;
    cl_float4 specular = {0, 0, 0, 0};
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, Texture texture, cl_float4 specular);
};

struct Mesh_idx {
    int vertices;
    int indices;
    int triangleNum;
    int texture_offset = -1;
    int textureIdx = -1;
    cl_float4 specular = {0, 0, 0, 0};
};

class Model {
public:
    /*  Методы   */
    Model(const std::string &path)
    {
        loadModel(path);
    }

    ~Model() {
        if (textures != nullptr) {
            delete textures;
            textures = nullptr;
        }
    }
    /*  Данные модели  */
    //std::vector<Mesh> meshes;
    std::vector<Mesh_idx> mesh_index;
    std::string directory;
    cl::Image2DArray * textures;
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
private:
    /*  Методы   */
    void loadModel(const std::string &path);
    void processNode(aiNode *node, const aiScene *scene, std::vector<Mesh> &meshes);
    Mesh processMesh(aiMesh *mesh, const aiScene *scene);
    Texture loadMaterialTexture(aiMaterial *mat, aiTextureType type);
};

#endif //GRAPHICS0_MESH_H
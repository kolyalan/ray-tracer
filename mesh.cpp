
#include "mesh.h"
#include <iostream>

extern cl::Context context;
extern cl::CommandQueue queue;

MyImage TextureFromFile(const char *path, const std::string &directory);

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, Texture texture, cl_float4 specular) {
    this->vertices = vertices;
    this->indices = indices;
    this->texture = texture;
    this->specular = specular;
}


void Model::loadModel(const std::string &path)
{
    Assimp::Importer import;
    const aiScene *scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);	
	
    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) 
    {
        std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
        return;
    }
    directory = path.substr(0, path.find_last_of('/'));
    std::vector<Mesh> meshes;
    processNode(scene->mRootNode, scene, meshes);
    mesh_index.resize(meshes.size());
    int nTextures = 0;
    int offset = 0;
    int width = 0, height = 0;
    for (int i = 0; i < meshes.size(); i++) {
        mesh_index[i].specular = meshes[i].specular;
        if (meshes[i].texture.hasTexture) {
            mesh_index[i].texture_offset = offset;
            mesh_index[i].textureIdx = nTextures;
            nTextures++;
            offset += meshes[i].texture.texture.getSize();
            if (width == 0 && height == 0) {
                width = meshes[i].texture.texture.getWidth();
                height = meshes[i].texture.texture.getHeight();
            }
            if (width != meshes[i].texture.texture.getWidth() || height != meshes[i].texture.texture.getHeight()) {
                width = std::max(width, meshes[i].texture.texture.getWidth());
                height = std::max(width, meshes[i].texture.texture.getHeight());
                std::cout << "Caution! Not all textures have equal sizes." << std::endl;
            }
        }
    }
    textures = new cl::Image2DArray(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_sRGBA, CL_UNORM_INT8), 
                                    nTextures+1, width, height, 0, 0);
    int vOff = 0, tOff = 0;
    for (int i = 0; i < meshes.size(); i++) {
        queue.enqueueWriteImage(*textures, CL_BLOCKING, {0, 0, (unsigned)mesh_index[i].textureIdx}, 
                                {(unsigned)width, (unsigned)height, 1}, 0, 0, meshes[i].texture.texture.getData());
        vertices.insert(vertices.end(), meshes[i].vertices.begin(), meshes[i].vertices.end());
        mesh_index[i].vertices = vOff;
        vOff = vertices.size();
        indices.insert(indices.end(), meshes[i].indices.begin(), meshes[i].indices.end());
        mesh_index[i].indices = tOff;
        tOff = indices.size() / 3;
        mesh_index[i].triangleNum = meshes[i].indices.size()/3;
    }
    MyImage noise = TextureFromFile("noise.png", directory);
    queue.enqueueWriteImage(*textures, CL_BLOCKING, {0, 0, (unsigned)nTextures}, 
                            {(unsigned)width, (unsigned)height, 1}, 0, 0, noise.getData());
}  

void Model::processNode(aiNode *node, const aiScene *scene, std::vector<Mesh> &meshes)
{
    // обработать все полигональные сетки в узле(если есть)
    for(unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]]; 
        meshes.push_back(processMesh(mesh, scene));			
    }
    // выполнить ту же обработку и для каждого потомка узла
    for(unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(node->mChildren[i], scene, meshes);
    }
}  

Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene)
{
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    Texture texture;
    cl_float4 specular;

    for(unsigned int i = 0; i < mesh->mNumVertices; i++)
    {
        Vertex vertex;
        // обработка координат, нормалей и текстурных координат вершин
        cl_float3 vector; 
        vector.s[0] = mesh->mVertices[i].x;
        vector.s[1] = mesh->mVertices[i].y;
        vector.s[2] = mesh->mVertices[i].z; 
        vertex.Position = vector;
        
        vector.s[0] = mesh->mNormals[i].x;
        vector.s[1] = mesh->mNormals[i].y;
        vector.s[2] = mesh->mNormals[i].z;
        vertex.Normal = vector;  

        if (mesh->mTextureCoords[0]) {// сетка обладает набором текстурных координат?
            cl_float2 vec;
            vec.s[0] = mesh->mTextureCoords[0][i].x; 
            vec.s[1] = mesh->mTextureCoords[0][i].y;
            vertex.TexCoords = vec;
        } else {
            vertex.TexCoords = {0.0f, 0.0f};  
        }
        vertices.push_back(vertex);
    }
    // орбаботка индексов
    for(unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        for(unsigned int j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }  

    // обработка материала
    if(mesh->mMaterialIndex >= 0) {
        aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];

        texture = loadMaterialTexture(material, aiTextureType_DIFFUSE);
        aiColor3D spec;
        material->Get(AI_MATKEY_COLOR_SPECULAR,spec);
        specular.s0 = spec.r;
        specular.s1 = spec.g;
        specular.s2 = spec.b;
        specular.s3 = 0;
    }  
    return Mesh(vertices, indices, texture, specular);
}  

Texture Model::loadMaterialTexture(aiMaterial *mat, aiTextureType type)
{
    Texture texture;
    if (mat->GetTextureCount(type) != 1) {
        std::cout << "Too many textures found" << std::endl;
    }
    aiString str;
    mat->GetTexture(type, 0, &str);
    texture = {TextureFromFile(str.C_Str(), directory), str.C_Str()};
    return texture;
}

MyImage TextureFromFile(const char *path, const std::string &directory) {
    std::string filename = std::string(path);
    filename = directory + '/' + filename;

    return MyImage(filename);
}
// Windows headers
#if defined(_WIN64)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#endif

// c++ standard lib headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

// GLM helpers for TRS/matrices
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(_WIN64)
#define STBI_MSC_SECURE_CRT
#endif
#include "tiny_gltf.h"

#if defined(_WIN64)
static inline void DebugOut(const std::string& s)
{
#ifdef UNICODE
    std::wstring ws(s.begin(), s.end());
    OutputDebugStringW(ws.c_str());
#else
    OutputDebugStringA(s.c_str());
#endif
}
#endif

#undef Log
#if defined(_WIN64)
#define Log(x) \
do { \
    std::stringstream s1765; s1765 << x; \
    printf("%s", s1765.str().c_str()); \
    DebugOut(s1765.str()); \
} while(0)
#else
#define Log(x) \
do { \
    std::stringstream s1765; s1765 << x; \
    LogFile(s1765.str().c_str()); \
    printf("%s", s1765.str().c_str()); \
} while(0)
#endif

using namespace std;
using namespace glm;

void extractVertexAttribute(const tinygltf::Model& model, const tinygltf::Primitive& primitive, const std::string& attributeName, std::vector<float>& outData, int& stride,
    std::vector<double>& min, std::vector<double>& max) {
    auto it = primitive.attributes.find(attributeName);
    if (it != primitive.attributes.end()) {
        const tinygltf::Accessor& accessor = model.accessors[it->second];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const float* bufferData = reinterpret_cast<const float*>(&(model.buffers[bufferView.buffer].data[accessor.byteOffset + bufferView.byteOffset]));
        stride = accessor.ByteStride(bufferView) ? (accessor.ByteStride(bufferView) / sizeof(float)) : tinygltf::GetNumComponentsInType(accessor.type);
        outData.assign(bufferData, bufferData + accessor.count * stride);
        for (double minVal : accessor.minValues) {
            min.push_back(minVal);
        }
        for (double maxVal : accessor.maxValues) {
            max.push_back(maxVal);
        }
    }
}

void extractIndexAttribute(const tinygltf::Model& model, const tinygltf::Primitive& primitive, std::vector<unsigned char>& outData, int& stride,
    std::vector<double>& min, std::vector<double>& max,
    tinygltf::Accessor& accessorOut, tinygltf::BufferView& bufferViewOut) {
    auto& accessor = model.accessors[primitive.indices];
    auto& bufferView = model.bufferViews[accessor.bufferView];
    const unsigned char* bufferData = reinterpret_cast<const unsigned char*>(&(model.buffers[bufferView.buffer].data[accessor.byteOffset + bufferView.byteOffset]));
    outData.assign(bufferData, bufferData + accessor.count * 4);
    for (double minVal : accessor.minValues) {
        min.push_back(minVal);
    }
    for (double maxVal : accessor.maxValues) {
        max.push_back(maxVal);
    }
    bufferViewOut.target = bufferView.target;
    accessorOut.componentType = accessor.componentType;
    accessorOut.count = accessor.count;
    accessorOut.type = accessor.type;
}

void importMesh(tinygltf::Model& destModel, const tinygltf::Model& srcModel, int meshIndex)
{
    const tinygltf::Mesh& srcMesh = srcModel.meshes[meshIndex];

    auto& primMesh = srcMesh.primitives[0];
    // Extract positions
    std::vector<float> positions;
    std::vector<double> posMin;
    std::vector<double> posMax;
    int posStrideMesh;
    extractVertexAttribute(srcModel, primMesh, "POSITION", positions, posStrideMesh, posMin, posMax);
    Log(" imported mesh buffer positions: " << positions.size() / posStrideMesh << std::endl);

    // Extract normals
    std::vector<float> normals;
    std::vector<double> normalMin;
    std::vector<double> normalMax;
    int normalStride;
    extractVertexAttribute(srcModel, primMesh, "NORMAL", normals, normalStride, normalMin, normalMax);
    Log(" imported mesh buffer normals: " << normals.size() / normalStride << std::endl);
    // Extract texture coordinates
    std::vector<float> texCoords;
    std::vector<double> texMin;
    std::vector<double> texMax;
    int texCoordStride;
    extractVertexAttribute(srcModel, primMesh, "TEXCOORD_0", texCoords, texCoordStride, texMin, texMax);
    Log(" imported mesh buffer tex 0: " << texCoords.size() / texCoordStride << std::endl);

    // indices
    tinygltf::Accessor indicesAccessor;
    tinygltf::BufferView indicesBufferView;
    // Extract normals
    std::vector<unsigned char> indices; // typeless buffer
    std::vector<double> indicesMin;
    std::vector<double> indicesMax;
    int indicesStride;
    extractIndexAttribute(srcModel, primMesh, indices, indicesStride, indicesMin, indicesMax, indicesAccessor, indicesBufferView);
    Log(" imported mesh buffer indices: " << indices.size() / 4 << std::endl);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        Log("Usage: GlbLodMergerApp <base_glb> <lod_dir>\n");
        return 0;
    }

    std::string baseFile = argv[1];
    std::string lodDir = argv[2];

    tinygltf::Model baseModel;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    // Load base GLB
    if (!loader.LoadBinaryFromFile(&baseModel, &err, &warn, baseFile)) {
        Log("Failed to load base GLB: " << err << "\n");
        return 0;
    }

    // Extract base name without extension
    std::string baseName = std::filesystem::path(baseFile).stem().string();

    // Keep track of indices of newly added meshes
    std::vector<int> newMeshIndices;

    // Find and merge LOD files
    for (int i = 1; i <= 1; ++i) {
        char lodFile[512];
        snprintf(lodFile, sizeof(lodFile), "%s/%s_%02d.glb", lodDir.c_str(), baseName.c_str(), i);
        if (!std::filesystem::exists(lodFile)) continue;

        tinygltf::Model lodModel;
        Log("Processing LOD GLB: " << lodFile << "\n");
        if (!loader.LoadBinaryFromFile(&lodModel, &err, &warn, lodFile)) {
            Log("Failed to load LOD GLB: " << lodFile << "\n");
            continue;
        }

        // Merge meshes from LOD model into base model
        for (int meshNum = 0; meshNum < int(lodModel.meshes.size()); ++meshNum) {
            importMesh(baseModel, lodModel, meshNum);
        }
        for (auto& mesh : lodModel.meshes) {
            tinygltf::Mesh newMesh = mesh;
        /*    for (auto& prim : newMesh.primitives) {
                // Copy indices accessor
                if (prim.indices >= 0) {
                    // Copy accessor
                    tinygltf::Accessor accessor = lodModel.accessors[prim.indices];
                    // Copy bufferView
                    int oldBufferViewIdx = accessor.bufferView;
                    tinygltf::BufferView bufferView = lodModel.bufferViews[oldBufferViewIdx];
                    // Copy buffer
                    int oldBufferIdx = bufferView.buffer;
                    tinygltf::Buffer buffer = lodModel.buffers[oldBufferIdx];

                    // Append buffer data
                    int newBufferIdx = int(baseModel.buffers.size());
                    baseModel.buffers.push_back(buffer);

                    // Update bufferView to point to new buffer
                    bufferView.buffer = newBufferIdx;
                    int newBufferViewIdx = int(baseModel.bufferViews.size());
                    baseModel.bufferViews.push_back(bufferView);

                    // Update accessor to point to new bufferView
                    accessor.bufferView = newBufferViewIdx;
                    int newAccessorIdx = int(baseModel.accessors.size());
                    baseModel.accessors.push_back(accessor);

                    // Update primitive to point to new accessor
                    prim.indices = newAccessorIdx;
                }

                // NOTE: If your LOD files have different vertex data, you must also copy attributes.
                // The following was intentionally left out in your current code. Uncomment and use if needed.
                // for (auto& attr : prim.attributes) {
                //     int attrAccessorIdx = attr.second;
                //     tinygltf::Accessor accessor = lodModel.accessors[attrAccessorIdx];
                //     int oldBufferViewIdx = accessor.bufferView;
                //     tinygltf::BufferView bufferView = lodModel.bufferViews[oldBufferViewIdx];
                //     int oldBufferIdx = bufferView.buffer;
                //     tinygltf::Buffer buffer = lodModel.buffers[oldBufferIdx];
                //     int newBufferIdx = int(baseModel.buffers.size());
                //     baseModel.buffers.push_back(buffer);
                //     bufferView.buffer = newBufferIdx;
                //     int newBufferViewIdx = int(baseModel.bufferViews.size());
                //     baseModel.bufferViews.push_back(bufferView);
                //     accessor.bufferView = newBufferViewIdx;
                //     int newAccessorIdx = int(baseModel.accessors.size());
                //     baseModel.accessors.push_back(accessor);
                //     attr.second = newAccessorIdx;
                // }
            }
            baseModel.meshes.push_back(newMesh);
            newMeshIndices.push_back(int(baseModel.meshes.size() - 1));
        */
        }
        
    }

    // Create a parent node for all newly added meshes and attach it to the scene
    if (!newMeshIndices.empty()) {
        // Determine target scene
        int sceneIndex = baseModel.defaultScene >= 0
            ? baseModel.defaultScene
            : (!baseModel.scenes.empty() ? 0 : -1);

        if (sceneIndex == -1) {
            tinygltf::Scene sc;
            sc.name = "Scene";
            sceneIndex = int(baseModel.scenes.size());
            baseModel.scenes.push_back(sc);
            baseModel.defaultScene = sceneIndex;
        }

        // Create LOD root node
        tinygltf::Node lodRoot;
        lodRoot.name = baseName + "_LODs";
        int lodRootIndex = int(baseModel.nodes.size());
        baseModel.nodes.push_back(lodRoot);
        baseModel.scenes[sceneIndex].nodes.push_back(lodRootIndex);

        // Create a child node per new mesh
        for (size_t k = 0; k < newMeshIndices.size(); ++k) {
            tinygltf::Node child;
            child.mesh = newMeshIndices[k];
            child.name = baseName + "_LOD_" + std::to_string(k);
            int childIndex = int(baseModel.nodes.size());
            baseModel.nodes.push_back(child);
            baseModel.nodes[lodRootIndex].children.push_back(childIndex);
        }
    }

    // Write merged GLB
    std::string outFile = baseFile.substr(0, baseFile.find_last_of('.')) + "_merged.glb";
    if (!loader.WriteGltfSceneToFile(&baseModel, outFile, true, true, true, true)) {
        Log("Failed to write merged GLB\n");
        return 0;
    }

    Log("Merged GLB written to: " << outFile << "\n");
    return 1;
}
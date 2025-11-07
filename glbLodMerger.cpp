// Windows headers
#if defined(_WIN64)
//#define NOMINMAX
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

static inline uint32_t ReadIndexAt(const std::vector<unsigned char>& idx, int stride, size_t i) {
    const size_t off = i * static_cast<size_t>(stride);
    uint32_t v = 0;
    switch (stride) {
    case 1: v = idx[off]; break;
    case 2: { uint16_t t; std::memcpy(&t, &idx[off], 2); v = t; break; }
    case 4: { uint32_t t; std::memcpy(&t, &idx[off], 4); v = t; break; }
    default: Log("Unsupported index stride: " << stride << "\n"); break;
    }
    return v;
}

// Replace your current extractVertexAttribute with this version to correctly handle KHR_mesh_quantization.
static inline float DecodeComponent(int componentType, bool normalized, const void* src)
{
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        float v;
        std::memcpy(&v, src, sizeof(float));
        return v;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        uint8_t v;
        std::memcpy(&v, src, 1);
        return normalized ? (float)v / 255.0f : (float)v;
    }
    case TINYGLTF_COMPONENT_TYPE_BYTE: {
        int8_t v;
        std::memcpy(&v, src, 1);
        if (normalized) {
            // Map to [-1,1]
            return std::max(-1.0f, (float)v / 127.0f);
        }
        return (float)v;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        uint16_t v;
        std::memcpy(&v, src, 2);
        return normalized ? (float)v / 65535.0f : (float)v;
    }
    case TINYGLTF_COMPONENT_TYPE_SHORT: {
        int16_t v;
        std::memcpy(&v, src, 2);
        if (normalized) {
            return std::max(-1.0f, (float)v / 32767.0f);
        }
        return (float)v;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
        uint32_t v;
        std::memcpy(&v, src, 4);
        return (float)v; // POSITION should not be UINT normally, but handle generically.
    }
    default:
        Log("Unsupported vertex componentType: " << componentType << "\n");
        return 0.0f;
    }
}

void extractVertexAttribute(const tinygltf::Model& model,
    const tinygltf::Primitive& primitive,
    const std::string& attributeName,
    std::vector<float>& outData,
    int& stride,
    std::vector<double>& min,
    std::vector<double>& max)
{
    outData.clear();
    stride = 0;
    min.clear();
    max.clear();

    auto it = primitive.attributes.find(attributeName);
    if (it == primitive.attributes.end()) return;

    const tinygltf::Accessor& accessor = model.accessors[it->second];
    if (accessor.bufferView < 0) {
        Log("Accessor bufferView < 0 for attribute " << attributeName << "\n");
        return;
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    const size_t numComponents = tinygltf::GetNumComponentsInType(accessor.type);
    if (numComponents == 0) {
        Log("Invalid accessor.type for attribute " << attributeName << "\n");
        return;
    }

    // Bytes per single component.
    int componentSize = 0;
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:          componentSize = 4; break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    case TINYGLTF_COMPONENT_TYPE_BYTE:           componentSize = 1; break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    case TINYGLTF_COMPONENT_TYPE_SHORT:          componentSize = 2; break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   componentSize = 4; break;
    default:
        Log("Unsupported vertex componentType: " << accessor.componentType << "\n");
        return;
    }

    // Byte stride between consecutive vertices in the buffer.
    const int byteStride = accessor.ByteStride(bufferView) > 0
        ? accessor.ByteStride(bufferView)
        : int(numComponents) * componentSize;

    if (byteStride < componentSize * int(numComponents)) {
        // Spec allows >= packed size. Anything smaller is invalid.
        Log("Invalid byteStride (" << byteStride << ") for attribute " << attributeName << "\n");
        return;
    }

    const size_t start = bufferView.byteOffset + accessor.byteOffset;
    // Last vertex must fit: start + (count-1)*byteStride + packedSize
    const size_t packedSize = numComponents * size_t(componentSize);
    const size_t lastByte = start + (accessor.count ? (accessor.count - 1) * size_t(byteStride) : 0) + packedSize;
    if (lastByte > buffer.data.size()) {
        Log("Buffer overrun risk while reading attribute " << attributeName << "\n");
        return;
    }

    const unsigned char* basePtr = reinterpret_cast<const unsigned char*>(buffer.data.data() + start);

    outData.resize(accessor.count * numComponents);
    stride = int(numComponents); // number of float components per vertex

    for (size_t i = 0; i < accessor.count; ++i) {
        const unsigned char* elem = basePtr + i * byteStride;
        // Components in the attribute are ALWAYS tightly packed starting at elem,
        // even if vertex is interleaved (extra bytes follow after the attribute data).
        for (size_t c = 0; c < numComponents; ++c) {
            const void* compSrc = elem + c * componentSize;
            outData[i * numComponents + c] = DecodeComponent(accessor.componentType, accessor.normalized, compSrc);
        }
    }

    // Use provided min/max if present.
    if (accessor.minValues.size() == numComponents) min = accessor.minValues;
    if (accessor.maxValues.size() == numComponents) max = accessor.maxValues;

    // Compute if missing.
    if (min.size() != numComponents || max.size() != numComponents) {
        min.assign(numComponents, std::numeric_limits<double>::infinity());
        max.assign(numComponents, -std::numeric_limits<double>::infinity());
        for (size_t i = 0; i < accessor.count; ++i) {
            const float* v = &outData[i * numComponents];
            for (size_t c = 0; c < numComponents; ++c) {
                double fv = (double)v[c];
                if (fv < min[c]) min[c] = fv;
                if (fv > max[c]) max[c] = fv;
            }
        }
    }
}

void extractIndexAttribute(const tinygltf::Model& model,
    const tinygltf::Primitive& primitive,
    std::vector<unsigned char>& outData,
    int& stride,
    std::vector<double>& min,
    std::vector<double>& max,
    tinygltf::Accessor& accessorOut,
    tinygltf::BufferView& bufferViewOut)
{
    if (primitive.indices < 0) {
        stride = 0;
        return;
    }

    const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    // Determine index element size
    int elementByteSize = 0;
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  elementByteSize = 1; break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: elementByteSize = 2; break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   elementByteSize = 4; break;
    default:
        Log("Unsupported index componentType: " << accessor.componentType << "\n");
        stride = 0;
        return;
    }
    stride = elementByteSize;

    // Compute start pointer & total bytes
    size_t byteOffset = bufferView.byteOffset + accessor.byteOffset;
    size_t totalBytes = accessor.count * static_cast<size_t>(elementByteSize);

    if (byteOffset + totalBytes > buffer.data.size()) {
        Log("Index accessor out of range (buffer overflow attempt)\n");
        stride = 0;
        return;
    }

    const unsigned char* src = reinterpret_cast<const unsigned char*>(buffer.data.data() + byteOffset);
    outData.assign(src, src + totalBytes);

    // Copy min/max if present; if absent, compute them
    if (!accessor.minValues.empty() && !accessor.maxValues.empty()) {
        min.insert(min.end(), accessor.minValues.begin(), accessor.minValues.end());
        max.insert(max.end(), accessor.maxValues.begin(), accessor.maxValues.end());
    } else {
        // Compute integer min/max
        uint64_t minVal = std::numeric_limits<uint64_t>::max();
        uint64_t maxVal = 0;
        if (elementByteSize == 1) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(src);
            for (size_t i = 0; i < accessor.count; ++i) {
                uint64_t v = p[i];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        } else if (elementByteSize == 2) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < accessor.count; ++i) {
                uint64_t v = p[i];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        } else { // 4
            const uint32_t* p = reinterpret_cast<const uint32_t*>(src);
            for (size_t i = 0; i < accessor.count; ++i) {
                uint64_t v = p[i];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }
        min.push_back(static_cast<double>(minVal));
        max.push_back(static_cast<double>(maxVal));
    }

    // Populate lightweight copies (only fields your later code reuses)
    bufferViewOut.target = bufferView.target;          // Should be ELEMENT_ARRAY_BUFFER (34963) for indices
    accessorOut.componentType = accessor.componentType;
    accessorOut.count = accessor.count;
    accessorOut.type = accessor.type; // Should be SCALAR
    accessorOut.bufferView = accessor.bufferView;
}
struct MeshImportData {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texCoords;
    std::vector<unsigned char> indices; // typeless buffer
    int posStride;
    int normalStride;
    int texCoordStride;
    int indicesStride;
    std::vector<double> posMin;
    std::vector<double> posMax;
};

void importMesh(MeshImportData& dest, const tinygltf::Model& srcModel, int meshIndex)
{
    const tinygltf::Mesh& srcMesh = srcModel.meshes[meshIndex];

    auto& primMesh = srcMesh.primitives[0];
    // Extract positions
    extractVertexAttribute(srcModel, primMesh, "POSITION", dest.positions, dest.posStride, dest.posMin, dest.posMax);
    Log(" imported mesh buffer positions: " << dest.positions.size() / dest.posStride << std::endl);

    // Extract normals
    std::vector<double> normalMin;
    std::vector<double> normalMax;
    extractVertexAttribute(srcModel, primMesh, "NORMAL", dest.normals, dest.normalStride, normalMin, normalMax);
    Log(" imported mesh buffer normals: " << dest.normals.size() / dest.normalStride << std::endl);
    // Extract texture coordinates
    std::vector<double> texMin;
    std::vector<double> texMax;
    extractVertexAttribute(srcModel, primMesh, "TEXCOORD_0", dest.texCoords, dest.texCoordStride, texMin, texMax);
    Log(" imported mesh buffer tex 0: " << dest.texCoords.size() / dest.texCoordStride << std::endl);

    // indices
    tinygltf::Accessor indicesAccessor;
    tinygltf::BufferView indicesBufferView;
    // Extract normals
    std::vector<double> indicesMin;
    std::vector<double> indicesMax;
    extractIndexAttribute(srcModel, primMesh, dest.indices, dest.indicesStride, indicesMin, indicesMax, indicesAccessor, indicesBufferView);
    Log(" imported mesh buffer indices: " << dest.indices.size() / dest.indicesStride << std::endl);
}

void validityCheck(MeshImportData& import, const tinygltf::Model& srcModel, int meshIndex)
{
    // Ensure primitive mode is TRIANGLES (or -1 which defaults to TRIANGLES)
    int primMode = srcModel.meshes[meshIndex].primitives[0].mode;
    if (primMode != -1 && primMode != TINYGLTF_MODE_TRIANGLES) {
        Log("ERROR: Mesh primitive mode is not TRIANGLES. Found mode: " << primMode << "\n");
        exit(1);
    }
    // iterate through triangles and check positions
    size_t indexCount = import.indices.size() / import.indicesStride;
    for (size_t i = 0; i < indexCount; ++i) {
        uint32_t idx = ReadIndexAt(import.indices, import.indicesStride, i);
        size_t posOffset = idx * import.posStride;
        if (posOffset + 2 >= import.positions.size()) {
            Log("ERROR: Index " << idx << " out of range for positions buffer\n");
            exit(1);
        }
        float x = import.positions[posOffset + 0];
        float y = import.positions[posOffset + 1];
        float z = import.positions[posOffset + 2];
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
            Log("ERROR: Invalid position at index " << idx << ": (" << x << ", " << y << ", " << z << ")\n");
            exit(1);
        }
        if (import.posMin.size() >= 3 && import.posMax.size() >= 3) {
            if (x < import.posMin[0] || x > import.posMax[0] ||
                y < import.posMin[1] || y > import.posMax[1] ||
                z < import.posMin[2] || z > import.posMax[2]) {
                Log("ERROR: Position at index " << idx << " out of bounds: (" << x << ", " << y << ", " << z << ")\n");
                exit(1);
            }
        }
    }
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
    Log("Processing BASE MODEL GLB: " << baseFile << "\n");
    if (!loader.LoadBinaryFromFile(&baseModel, &err, &warn, baseFile)) {
        Log("Failed to load base GLB: " << err << "\n");
        return 0;
    }
    MeshImportData baseMeshData;
    importMesh(baseMeshData, baseModel, 0);
    validityCheck(baseMeshData, baseModel, 0);
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
            MeshImportData lodMeshData;
            importMesh(lodMeshData, lodModel, meshNum);
            validityCheck(lodMeshData, lodModel, meshNum);
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
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
#include <cstring>

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

void createLodMesh(tinygltf::Model& ioModel,
                   tinygltf::Mesh& outMesh,
                   const MeshImportData& lodMeshData,
                   int baseMeshIndex = 0,
                   int basePrimIndex = 0)
{
    outMesh = tinygltf::Mesh();
    outMesh.name = "LOD_Mesh";

    // Validate there is position data
    if (lodMeshData.posStride <= 0 || lodMeshData.positions.empty()) {
        Log("createLodMesh: LOD has no POSITION data\n");
        return;
    }

    // We optionally reuse base material for visual parity.
    int reuseMaterial = -1;
    if (baseMeshIndex >= 0 && baseMeshIndex < static_cast<int>(ioModel.meshes.size())) {
        const tinygltf::Mesh& baseMesh = ioModel.meshes[baseMeshIndex];
        if (basePrimIndex >= 0 && basePrimIndex < static_cast<int>(baseMesh.primitives.size())) {
            reuseMaterial = baseMesh.primitives[basePrimIndex].material;
        }
    }

    // Prepare a single buffer and pack: positions [ + normals ] [ + uvs ] + indices (aligned to 4 bytes).
    auto align4 = [](size_t v) -> size_t { return (v + 3u) & ~size_t(3u); };

    std::vector<unsigned char> bin;
    bin.reserve(
        lodMeshData.positions.size() * sizeof(float) +
        lodMeshData.normals.size()   * sizeof(float) +
        lodMeshData.texCoords.size() * sizeof(float) +
        lodMeshData.indices.size() + 16);

    // Write positions (float)
    size_t posOffset = align4(bin.size());
    bin.resize(posOffset);
    {
        const unsigned char* src = reinterpret_cast<const unsigned char*>(lodMeshData.positions.data());
        bin.insert(bin.end(), src, src + lodMeshData.positions.size() * sizeof(float));
    }
    const size_t posByteLength = lodMeshData.positions.size() * sizeof(float);

    // Optionally write normals if count matches positions
    const size_t posCount = lodMeshData.positions.size() / static_cast<size_t>(lodMeshData.posStride);
    int normalAccessorIndex = -1;
    size_t normalOffset = 0, normalByteLength = 0;
    if (lodMeshData.normalStride > 0 && !lodMeshData.normals.empty()) {
        const size_t normalCount = lodMeshData.normals.size() / static_cast<size_t>(lodMeshData.normalStride);
        if (normalCount == posCount && lodMeshData.normalStride == 3) {
            normalOffset = align4(bin.size());
            bin.resize(normalOffset);
            const unsigned char* src = reinterpret_cast<const unsigned char*>(lodMeshData.normals.data());
            bin.insert(bin.end(), src, src + lodMeshData.normals.size() * sizeof(float));
            normalByteLength = lodMeshData.normals.size() * sizeof(float);
        } else {
            Log("createLodMesh: Skipping NORMALS (count/stride mismatch)\n");
        }
    }

    // Optionally write UV0 if count matches positions
    int uv0AccessorIndex = -1;
    size_t uv0Offset = 0, uv0ByteLength = 0;
    if (lodMeshData.texCoordStride > 0 && !lodMeshData.texCoords.empty()) {
        const size_t uvCount = lodMeshData.texCoords.size() / static_cast<size_t>(lodMeshData.texCoordStride);
        if (uvCount == posCount && (lodMeshData.texCoordStride == 2 || lodMeshData.texCoordStride == 3)) {
            uv0Offset = align4(bin.size());
            bin.resize(uv0Offset);
            const unsigned char* src = reinterpret_cast<const unsigned char*>(lodMeshData.texCoords.data());
            bin.insert(bin.end(), src, src + lodMeshData.texCoords.size() * sizeof(float));
            uv0ByteLength = lodMeshData.texCoords.size() * sizeof(float);
        } else {
            Log("createLodMesh: Skipping TEXCOORD_0 (count/stride mismatch)\n");
        }
    }

    // Write indices (typeless buffer already correctly packed with stride 1/2/4)
    if (lodMeshData.indicesStride != 1 &&
        lodMeshData.indicesStride != 2 &&
        lodMeshData.indicesStride != 4) {
        Log("createLodMesh: unsupported indices stride " << lodMeshData.indicesStride << "\n");
        return;
    }
    const size_t indexCount = lodMeshData.indices.size() / static_cast<size_t>(lodMeshData.indicesStride);
    if (indexCount == 0) {
        Log("createLodMesh: LOD indices are empty\n");
        return;
    }
    size_t idxOffset = align4(bin.size());
    bin.resize(idxOffset);
    bin.insert(bin.end(), lodMeshData.indices.begin(), lodMeshData.indices.end());
    const size_t idxByteLength = lodMeshData.indices.size();

    // Create one buffer for all data
    tinygltf::Buffer buffer;
    buffer.data = std::move(bin);
    const int bufferIndex = static_cast<int>(ioModel.buffers.size());
    ioModel.buffers.push_back(std::move(buffer));

    // Create BufferViews
    // Positions
    tinygltf::BufferView posBV;
    posBV.name = "LOD_Positions";
    posBV.buffer = bufferIndex;
    posBV.byteOffset = posOffset;
    posBV.byteLength = posByteLength;
    posBV.byteStride = 0; // tightly packed float[N]
    posBV.target = 34962; // ARRAY_BUFFER
    const int posBVIndex = static_cast<int>(ioModel.bufferViews.size());
    ioModel.bufferViews.push_back(std::move(posBV));

    // Normals (optional)
    int normalBVIndex = -1;
    if (normalByteLength > 0) {
        tinygltf::BufferView bv;
        bv.name = "LOD_Normals";
        bv.buffer = bufferIndex;
        bv.byteOffset = normalOffset;
        bv.byteLength = normalByteLength;
        bv.byteStride = 0;
        bv.target = 34962; // ARRAY_BUFFER
        normalBVIndex = static_cast<int>(ioModel.bufferViews.size());
        ioModel.bufferViews.push_back(std::move(bv));
    }

    // UV0 (optional)
    int uv0BVIndex = -1;
    if (uv0ByteLength > 0) {
        tinygltf::BufferView bv;
        bv.name = "LOD_Texcoord0";
        bv.buffer = bufferIndex;
        bv.byteOffset = uv0Offset;
        bv.byteLength = uv0ByteLength;
        bv.byteStride = 0;
        bv.target = 34962; // ARRAY_BUFFER
        uv0BVIndex = static_cast<int>(ioModel.bufferViews.size());
        ioModel.bufferViews.push_back(std::move(bv));
    }

    // Indices
    tinygltf::BufferView idxBV;
    idxBV.name = "LOD_Indices";
    idxBV.buffer = bufferIndex;
    idxBV.byteOffset = idxOffset;
    idxBV.byteLength = idxByteLength;
    idxBV.byteStride = 0; // tightly packed
    idxBV.target = 34963; // ELEMENT_ARRAY_BUFFER
    const int idxBVIndex = static_cast<int>(ioModel.bufferViews.size());
    ioModel.bufferViews.push_back(std::move(idxBV));

    // Accessors
    // Positions
    tinygltf::Accessor posAccessor;
    posAccessor.name = "LOD_PositionAccessor";
    posAccessor.bufferView = posBVIndex;
    posAccessor.byteOffset = 0;
    posAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    posAccessor.count = posCount;
    posAccessor.type = (lodMeshData.posStride == 2) ? TINYGLTF_TYPE_VEC2
                     : (lodMeshData.posStride == 4) ? TINYGLTF_TYPE_VEC4
                     : TINYGLTF_TYPE_VEC3; // default
    posAccessor.normalized = false;
    if (lodMeshData.posMin.size() >= static_cast<size_t>(lodMeshData.posStride) &&
        lodMeshData.posMax.size() >= static_cast<size_t>(lodMeshData.posStride)) {
        posAccessor.minValues.assign(lodMeshData.posMin.begin(),
                                     lodMeshData.posMin.begin() + lodMeshData.posStride);
        posAccessor.maxValues.assign(lodMeshData.posMax.begin(),
                                     lodMeshData.posMax.begin() + lodMeshData.posStride);
    }
    const int posAccessorIndex = static_cast<int>(ioModel.accessors.size());
    ioModel.accessors.push_back(std::move(posAccessor));

    // Normals (optional, only if present and valid)
    if (normalByteLength > 0) {
        tinygltf::Accessor n;
        n.name = "LOD_NormalAccessor";
        n.bufferView = normalBVIndex;
        n.byteOffset = 0;
        n.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        n.count = posCount; // matched above
        n.type = TINYGLTF_TYPE_VEC3;
        n.normalized = false;
        normalAccessorIndex = static_cast<int>(ioModel.accessors.size());
        ioModel.accessors.push_back(std::move(n));
    }

    // UV0 (optional, only if present and valid)
    if (uv0ByteLength > 0) {
        tinygltf::Accessor uv;
        uv.name = "LOD_UV0Accessor";
        uv.bufferView = uv0BVIndex;
        uv.byteOffset = 0;
        uv.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        uv.count = posCount; // matched above
        uv.type = (lodMeshData.texCoordStride == 3) ? TINYGLTF_TYPE_VEC3 : TINYGLTF_TYPE_VEC2;
        uv.normalized = false;
        uv0AccessorIndex = static_cast<int>(ioModel.accessors.size());
        ioModel.accessors.push_back(std::move(uv));
    }

    // Indices
    int indexComponentType = -1;
    switch (lodMeshData.indicesStride) {
        case 1: indexComponentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;  break;
        case 2: indexComponentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT; break;
        case 4: indexComponentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;   break;
    }
    tinygltf::Accessor idxAccessor;
    idxAccessor.name = "LOD_IndexAccessor";
    idxAccessor.bufferView = idxBVIndex;
    idxAccessor.byteOffset = 0;
    idxAccessor.componentType = indexComponentType;
    idxAccessor.count = indexCount;
    idxAccessor.type = TINYGLTF_TYPE_SCALAR;
    idxAccessor.normalized = false;
    // Fill min/max for indices (optional)
    {
        uint64_t minIdx = std::numeric_limits<uint64_t>::max();
        uint64_t maxIdx = 0;
        for (size_t i = 0; i < indexCount; ++i) {
            uint32_t v = ReadIndexAt(lodMeshData.indices, lodMeshData.indicesStride, i);
            if (v < minIdx) minIdx = v;
            if (v > maxIdx) maxIdx = v;
        }
        idxAccessor.minValues = { static_cast<double>(minIdx) };
        idxAccessor.maxValues = { static_cast<double>(maxIdx) };
    }
    const int idxAccessorIndex = static_cast<int>(ioModel.accessors.size());
    ioModel.accessors.push_back(std::move(idxAccessor));

    // Assemble primitive with the freshly built accessors
    tinygltf::Primitive prim;
    prim.mode = TINYGLTF_MODE_TRIANGLES;
    prim.indices = idxAccessorIndex;
    prim.attributes["POSITION"] = posAccessorIndex;
    if (normalAccessorIndex >= 0) {
        prim.attributes["NORMAL"] = normalAccessorIndex;
    }
    if (uv0AccessorIndex >= 0) {
        prim.attributes["TEXCOORD_0"] = uv0AccessorIndex;
    }
    prim.material = reuseMaterial;

    outMesh.primitives.push_back(std::move(prim));
}

static void AttachLodsUnderBaseNodes(tinygltf::Model& model,
                                     const std::vector<int>& lodMeshIndices,
                                     int baseMeshIndex,
                                     const std::string& baseName)
{
    if (lodMeshIndices.empty()) return;

    // Find nodes that reference the base mesh.
    std::vector<int> baseNodeIndices;
    for (int ni = 0; ni < static_cast<int>(model.nodes.size()); ++ni) {
        if (model.nodes[ni].mesh == baseMeshIndex) {
            baseNodeIndices.push_back(ni);
        }
    }

    if (baseNodeIndices.empty()) {
        Log("WARNING: No node references base mesh " << baseMeshIndex
            << ". Cannot attach LODs under base transform; keeping prior layout.\n");
        return;
    }

    for (int baseNodeIdx : baseNodeIndices) {
        // Create a LOD group under the base node (inherits base transform).
        tinygltf::Node lodGroup;
        lodGroup.name = baseName + "_LODs";
        const int lodGroupIndex = static_cast<int>(model.nodes.size());
        model.nodes.push_back(lodGroup);
        model.nodes[baseNodeIdx].children.push_back(lodGroupIndex);

        // Create one child per LOD mesh, identity TRS (inherit parent's transform).
        for (size_t k = 0; k < lodMeshIndices.size(); ++k) {
            tinygltf::Node child;
            child.mesh = lodMeshIndices[k];
            child.name = baseName + "_LOD_" + std::to_string(k);
            const int childIndex = static_cast<int>(model.nodes.size());
            model.nodes.push_back(child);
            model.nodes[lodGroupIndex].children.push_back(childIndex);
        }
    }
}

// Add after includes and before your existing helpers.

// Merge extensionsUsed from src into dst
static void MergeExtensionsUsed(tinygltf::Model& dst, const tinygltf::Model& src) {
    for (const auto& e : src.extensionsUsed) {
        if (std::find(dst.extensionsUsed.begin(), dst.extensionsUsed.end(), e) == dst.extensionsUsed.end()) {
            dst.extensionsUsed.push_back(e);
        }
    }
}

struct CloneContext {
    std::vector<int> bufferMap;
    std::vector<int> bufferViewMap;
    std::vector<int> accessorMap;
    std::vector<int> imageMap;
    std::vector<int> textureMap;
    std::vector<int> samplerMap;
    std::vector<int> materialMap;
};

static int MapBuffer(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.bufferMap.size() <= idx) ctx.bufferMap.resize(idx + 1, -1);
    if (ctx.bufferMap[idx] >= 0) return ctx.bufferMap[idx];
    tinygltf::Buffer b = src.buffers[idx];
    int newIdx = (int)dst.buffers.size();
    dst.buffers.push_back(std::move(b));
    ctx.bufferMap[idx] = newIdx;
    return newIdx;
}

static int MapBufferView(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.bufferViewMap.size() <= idx) ctx.bufferViewMap.resize(idx + 1, -1);
    if (ctx.bufferViewMap[idx] >= 0) return ctx.bufferViewMap[idx];
    tinygltf::BufferView bv = src.bufferViews[idx];
    bv.buffer = MapBuffer(ctx, src, bv.buffer, dst);
    int newIdx = (int)dst.bufferViews.size();
    dst.bufferViews.push_back(std::move(bv));
    ctx.bufferViewMap[idx] = newIdx;
    return newIdx;
}

static int MapAccessor(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.accessorMap.size() <= idx) ctx.accessorMap.resize(idx + 1, -1);
    if (ctx.accessorMap[idx] >= 0) return ctx.accessorMap[idx];
    tinygltf::Accessor a = src.accessors[idx];
    a.bufferView = MapBufferView(ctx, src, a.bufferView, dst);
    // Map sparse data if present
    if (a.sparse.isSparse) {
        a.sparse.indices.bufferView = MapBufferView(ctx, src, a.sparse.indices.bufferView, dst);
        a.sparse.values.bufferView  = MapBufferView(ctx, src, a.sparse.values.bufferView, dst);
    }
    int newIdx = (int)dst.accessors.size();
    dst.accessors.push_back(std::move(a));
    ctx.accessorMap[idx] = newIdx;
    return newIdx;
}

static int MapSampler(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.samplerMap.size() <= idx) ctx.samplerMap.resize(idx + 1, -1);
    if (ctx.samplerMap[idx] >= 0) return ctx.samplerMap[idx];
    tinygltf::Sampler s = src.samplers[idx];
    int newIdx = (int)dst.samplers.size();
    dst.samplers.push_back(std::move(s));
    ctx.samplerMap[idx] = newIdx;
    return newIdx;
}

static int MapImage(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.imageMap.size() <= idx) ctx.imageMap.resize(idx + 1, -1);
    if (ctx.imageMap[idx] >= 0) return ctx.imageMap[idx];
    tinygltf::Image img = src.images[idx];
    if (img.bufferView >= 0) {
        img.bufferView = MapBufferView(ctx, src, img.bufferView, dst);
        // img.mimeType left as-is
    }
    int newIdx = (int)dst.images.size();
    dst.images.push_back(std::move(img));
    ctx.imageMap[idx] = newIdx;
    return newIdx;
}

static int MapTexture(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.textureMap.size() <= idx) ctx.textureMap.resize(idx + 1, -1);
    if (ctx.textureMap[idx] >= 0) return ctx.textureMap[idx];
    tinygltf::Texture t = src.textures[idx];
    t.source  = MapImage(ctx, src, t.source, dst);
    t.sampler = MapSampler(ctx, src, t.sampler, dst);
    int newIdx = (int)dst.textures.size();
    dst.textures.push_back(std::move(t));
    ctx.textureMap[idx] = newIdx;
    return newIdx;
}

static void RemapTexInfo(CloneContext& ctx, const tinygltf::Model& src, tinygltf::TextureInfo& ti, tinygltf::Model& dst) {
    ti.index = MapTexture(ctx, src, ti.index, dst);
}
static void RemapOccl(CloneContext& ctx, const tinygltf::Model& src, tinygltf::OcclusionTextureInfo& oi, tinygltf::Model& dst) {
    oi.index = MapTexture(ctx, src, oi.index, dst);
}
static void RemapNorm(CloneContext& ctx, const tinygltf::Model& src, tinygltf::NormalTextureInfo& ni, tinygltf::Model& dst) {
    ni.index = MapTexture(ctx, src, ni.index, dst);
}

static int MapMaterial(CloneContext& ctx, const tinygltf::Model& src, int idx, tinygltf::Model& dst) {
    if (idx < 0) return -1;
    if ((int)ctx.materialMap.size() <= idx) ctx.materialMap.resize(idx + 1, -1);
    if (ctx.materialMap[idx] >= 0) return ctx.materialMap[idx];
    tinygltf::Material m = src.materials[idx];
    RemapTexInfo(ctx, src, m.pbrMetallicRoughness.baseColorTexture, dst);
    RemapTexInfo(ctx, src, m.pbrMetallicRoughness.metallicRoughnessTexture, dst);
    RemapNorm(ctx, src, m.normalTexture, dst);
    RemapOccl(ctx, src, m.occlusionTexture, dst);
    RemapTexInfo(ctx, src, m.emissiveTexture, dst);
    int newIdx = (int)dst.materials.size();
    dst.materials.push_back(std::move(m));
    ctx.materialMap[idx] = newIdx;
    return newIdx;
}

// Clone one mesh from src into dst, returning new mesh index in dst.
static int CloneMeshWithDependencies(CloneContext& ctx,
                                     const tinygltf::Model& src,
                                     int srcMeshIndex,
                                     tinygltf::Model& dst)
{
    if (srcMeshIndex < 0 || srcMeshIndex >= (int)src.meshes.size()) return -1;

    tinygltf::Mesh out = src.meshes[srcMeshIndex];
    for (auto& prim : out.primitives) {
        // Remap accessors for attributes
        for (auto& kv : prim.attributes) {
            kv.second = MapAccessor(ctx, src, kv.second, dst);
        }
        // Indices
        prim.indices = MapAccessor(ctx, src, prim.indices, dst);
        // Material
        prim.material = MapMaterial(ctx, src, prim.material, dst);
        // Morph targets
        for (auto& tgt : prim.targets) {
            for (auto& kv : tgt) kv.second = MapAccessor(ctx, src, kv.second, dst);
        }
    }
    int newIdx = (int)dst.meshes.size();
    dst.meshes.push_back(std::move(out));

    // Merge extensionsUsed
    MergeExtensionsUsed(dst, src);
    return newIdx;
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
    // if we start with a _00 suffix, remove it
    if (baseName.length() > 3 &&
        baseName[baseName.length() - 3] == '_' &&
        isdigit(baseName[baseName.length() - 2]) &&
        isdigit(baseName[baseName.length() - 1])) {
        baseName = baseName.substr(0, baseName.length() - 3);
    }

    // Keep track of indices of newly added meshes
    std::vector<int> newMeshIndices;

    // Find and merge LOD files
    for (int i = 1; i <= 9; ++i) {
        char lodFile[512];
        snprintf(lodFile, sizeof(lodFile), "%s/%s_%02d.glb", lodDir.c_str(), baseName.c_str(), i);
        if (!std::filesystem::exists(lodFile)) continue;

        tinygltf::Model lodModel;
        Log("Processing LOD GLB: " << lodFile << "\n");
        if (!loader.LoadBinaryFromFile(&lodModel, &err, &warn, lodFile)) {
            Log("Failed to load LOD GLB: " << lodFile << "\n");
            continue;
        }

        bool clone = false;
        if (clone) {
            // Merge meshes from LOD model into base model by cloning, not re-importing.
            CloneContext ctx; // persist per LOD file to dedup shared buffers/materials
            for (int meshNum = 0; meshNum < (int)lodModel.meshes.size(); ++meshNum) {
                int newIdx = CloneMeshWithDependencies(ctx, lodModel, meshNum, baseModel);
                if (newIdx >= 0) {
                    newMeshIndices.push_back(newIdx);
                }
            }
        } else {
            // Merge meshes from LOD model into base model
            for (int meshNum = 0; meshNum < int(lodModel.meshes.size()); ++meshNum) {
                MeshImportData lodMeshData;
                importMesh(lodMeshData, lodModel, meshNum);
                validityCheck(lodMeshData, lodModel, meshNum);
                tinygltf::Mesh newLodMesh;
                createLodMesh(baseModel, newLodMesh, lodMeshData, /*baseMeshIndex=*/0, /*basePrimIndex=*/0);
                if (!newLodMesh.primitives.empty()) {
                    baseModel.meshes.push_back(newLodMesh);
                    newMeshIndices.push_back(static_cast<int>(baseModel.meshes.size() - 1));
                }
            }
        }
        
    }

    // Attach LOD meshes under the base mesh node(s) so transforms match.
    if (!newMeshIndices.empty()) {
        const int baseMeshIndex = 0; // adjust if your base mesh index differs
        AttachLodsUnderBaseNodes(baseModel, newMeshIndices, baseMeshIndex, baseName);
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
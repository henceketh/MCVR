#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/build_geometry.hpp"
#include "core/render/emission.hpp"
#include "core/render/world.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Framework;

struct PrebuiltEntityGeometry;

struct EntitiesBuildTask {
    float lineWidth;
    World::Coordinates coordinate;
    bool normalOffset;
    int entityCount;
    int64_t *entityKeys;
    double *entityXs;
    double *entityYs;
    double *entityZs;
    int *entityRayTracingFlags;
    int *entityPostRenderFlags;
    int *entityPrebuiltBLASs;
    const float *entityTransforms = nullptr;
    const vk::VertexFormat::EntityInstanceMaterial *entityMaterials = nullptr;
    int *entityPosts;
    int *entityGeometryCounts;
    int *geometryTypes;
    const char **geometryGroupNames;
    const char **geometryContentNames;
    int *geometryTextures;
    float *geometryNormalOffsets;
    int *vertexFormats;
    int *indexFormats;
    int *vertexCounts;
    void **vertices;
    std::shared_ptr<BuildBufferLease> transferredBufferLease;
};

struct EntityBuildInput : public SharedObject<EntityBuildInput> {
    int64_t entityKey;
    double x, y, z;
    int rayTracingFlag;
    int postRenderFlag;
    int prebuiltBLAS;
    World::Coordinates coordinate;
    glm::mat4 localTransform{1.0f};
    vk::VertexFormat::EntityInstanceMaterial material{};
    std::shared_ptr<PrebuiltEntityGeometry> sharedGeometry;
    std::shared_ptr<std::array<float, 260>> attributeTransform;
    std::vector<std::array<float, 16>> programVertices;
    bool indexedTriangles = false;
    uint32_t geometryCount;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::string> geometryGroupNames;
    std::vector<std::string> geometryContentNames;
    std::vector<BuildGeometry> geometries;
    std::vector<LightInfo> lightInfos;
    std::vector<VkDeviceAddress> indexBufferAddresses;
    std::vector<VkDeviceAddress> pbrBufferAddresses;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<vk::BLASBuilder> blasBuilder;

    EntityBuildInput(int64_t entityKey,
                     double x,
                     double y,
                     double z,
                     int rayTracingFlag,
                     int postRenderFlag,
                     int prebuiltBLAS,
                     World::Coordinates coordinate,
                     uint32_t geometryCount,
                     std::vector<World::GeometryTypes> &&geometryTypes,
                     std::vector<std::string> &&geometryGroupNames,
                     std::vector<std::string> &&geometryContentNames,
                     std::vector<BuildGeometry> &&geometries);

    void buildGeometryResources(std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder,
                                std::shared_ptr<vk::DeviceLocalBuffer> pbrCacheBuffer,
                                VkDeviceSize pbrVertexOffset,
                                std::shared_ptr<vk::DeviceLocalBuffer> indexCacheBuffer,
                                VkDeviceSize indexOffset);
    void buildLightInfos(const Emission &emission, bool indexedTriangles = false, const EntityBuildInput *model = nullptr);
    void releaseGeometryInput();
};

struct EntityPostBuildBatch : public SharedObject<EntityPostBuildBatch> {
    std::vector<std::shared_ptr<EntityBuildInput>> buildInputs;

    void addBuildInput(std::shared_ptr<EntityBuildInput> buildInput);
};

struct Entity;
struct EntityBatch;
struct EntityPost;
struct EntityPostBatch;

struct Entity : public SharedObject<Entity> {
    int64_t entityKey = 0;
    double x, y, z;
    int rayTracingFlag;
    int prebuiltBLAS = -1;
    World::Coordinates coordinate;

    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<std::vector<VkDeviceAddress>> indexBufferAddresses;
    std::shared_ptr<std::vector<VkDeviceAddress>> pbrBufferAddresses;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;

    uint32_t geometryCount;
    std::shared_ptr<std::vector<std::string>> geometryGroupNames;
    std::shared_ptr<std::vector<std::string>> geometryContentNames;
    std::shared_ptr<std::vector<uint32_t>> vertexCounts;
    std::shared_ptr<std::vector<uint32_t>> indexCounts;

    glm::mat4 localTransform{1.0f};
    vk::VertexFormat::EntityInstanceMaterial material{};
    std::shared_ptr<PrebuiltEntityGeometry> sharedGeometry;
    std::shared_ptr<std::array<float, 260>> attributeTransform;
    uint64_t lastQueuedFrame = 0;
    uint64_t lastActiveFrame = std::numeric_limits<uint64_t>::max();
    bool hasSavedRenderState = false;
    std::shared_ptr<PrebuiltEntityGeometry> lastFrameSharedGeometry;
    std::shared_ptr<std::vector<VkDeviceAddress>> lastFramePbrAddressSource;
    VkTransformMatrixKHR lastFrameTransform{};
    std::vector<VkDeviceAddress> lastFrameIndexBufferAddresses;
    std::vector<VkDeviceAddress> lastFramePbrBufferAddresses;
    std::vector<std::string> lastFrameGeometryGroupNames;
    std::vector<std::string> lastFrameGeometryContentNames;
    std::vector<uint32_t> lastFrameVertexCounts;
    std::vector<uint32_t> lastFrameIndexCounts;

    Entity() = default;
    void updateRenderMetadata(std::shared_ptr<EntityBuildInput> buildInput);
    void updateGeometryResources(std::shared_ptr<EntityBuildInput> buildInput);
    void saveRenderStateForNextFrame(VkTransformMatrixKHR transform);
};

struct EntityBatch : public SharedObject<EntityBatch> {
    std::vector<std::shared_ptr<Entity>> entities;

    EntityBatch(std::vector<std::shared_ptr<Entity>> &&entities);
};

struct EntityPost : public SharedObject<EntityPost> {
    int postRenderFlag;
    double x, y, z;

    uint32_t geometryCount;
    std::vector<std::string> geometryContentNames;
    std::vector<uint32_t> indexCounts;

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> vertexBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indexBuffers;

    EntityPost(std::shared_ptr<EntityBuildInput> buildInput);
};

struct EntityPostBatch : public SharedObject<EntityPostBatch> {
    std::vector<std::shared_ptr<EntityPost>> entities;

    EntityPostBatch(std::shared_ptr<EntityPostBuildBatch> postBuildBatch);
};

class Entities : public SharedObject<Entities> {
    friend World;

  public:
    class VertexShaderEvaluator {
      public:
        static constexpr uint32_t channelCount = 16;
        static constexpr uint32_t vertexBinding = 16;
        static constexpr uint32_t outputBinding = 17;
        struct Scalar {
            std::array<float, channelCount + 1> coefficients{};
            uint32_t bits = 0;
            float lower = 0, upper = 0;
            bool affine = true;
            bool uniform = true;

            static Scalar number(float value) {
                Scalar s;
                s.coefficients[channelCount] = value;
                s.bits = std::bit_cast<uint32_t>(value);
                s.lower = s.upper = value;
                return s;
            }
            static Scalar integer(uint32_t value) {
                Scalar s;
                s.bits = value;
                s.coefficients[channelCount] = s.lower = s.upper = static_cast<float>(value);
                return s;
            }
            static Scalar input(uint32_t channel, float lower, float upper);
            float constant() const { return coefficients[channelCount]; }
        };
        using Vertex = std::array<Scalar, channelCount>;
        struct Result {
            Vertex vertex;
            bool completed = false;
            bool affinePosition = false;
            bool affineAttributes = false;
            uint64_t attributeRevision = 0;
            std::string error;
#if defined(MCVR_ENABLE_PROFILER)
            uint32_t executedInstructions = 0;
            uint32_t checkedInputInstructions = 0;
            uint32_t skippedCommonInputInstructions = 0;
            bool scheduled = false;
#endif
        };
        using Bindings = std::array<std::span<const uint8_t>, 32>;

        class InstanceBatch {
          public:
            InstanceBatch(const VertexShaderEvaluator &program, const Bindings &bindings, const Vertex &bounds);
            Result evaluate(std::span<const uint8_t> instance);
          private:
            const VertexShaderEvaluator &program_;
            Bindings bindings_;
            Vertex bounds_;
            uint64_t serial_;
        };

        explicit VertexShaderEvaluator(const std::string &source);
        ~VertexShaderEvaluator();
        VertexShaderEvaluator(VertexShaderEvaluator &&) noexcept;
        VertexShaderEvaluator &operator=(VertexShaderEvaluator &&) noexcept;
        Result evaluate(const Bindings &bindings, const Vertex &vertex, uint32_t vertexIndex = 0) const;
        const std::vector<uint32_t> &spirv() const;
        std::vector<uint8_t> inputKey(const Bindings &bindings) const;
        void inputKey(const Bindings &bindings, std::vector<uint8_t> &key,
                      uint32_t bindingMask = std::numeric_limits<uint32_t>::max()) const;
        void instanceOrderKey(std::span<const uint8_t> data, std::vector<uint8_t> &key) const;

      private:
        Result evaluateInternal(const Bindings &bindings, const Vertex &vertex, uint32_t vertexIndex,
                                uint64_t batchSerial) const;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    struct VertexShaderInstance {
        int64_t key, owner;
        int geometry, program;
        double x, y, z;
        glm::mat4 pose;
        glm::mat3 normal;
        std::span<const uint8_t> data;
    };

    Entities(std::shared_ptr<Framework> framework);

    void resetFrame();
    void queueBuild(EntitiesBuildTask task);
    int registerPrebuiltGeometry(std::shared_ptr<EntityBuildInput> input, int previous = -1);
    void releasePrebuiltGeometry(int handle);
    int registerVertexShader(const std::string &source);
    void releaseVertexShader(int handle);
    std::vector<std::pair<int64_t, std::string>> queueVertexShaderInstances(
        const std::vector<VertexShaderInstance> &instances, const VertexShaderEvaluator::Bindings &uniforms);
    bool submitPendingBLAS(std::shared_ptr<vk::CommandBuffer> commandBuffer);
    void build();
    void close();
    std::shared_ptr<EntityBatch> entityBatch();
    std::shared_ptr<EntityBatch> peekEntityBatch() const;
    std::shared_ptr<EntityPostBatch> entityPostBatch();

  private:
    std::unordered_map<int, std::shared_ptr<PrebuiltEntityGeometry>> prebuiltGeometries_;
    std::unordered_map<int, std::shared_ptr<VertexShaderEvaluator>> vertexShaders_;
    struct ShaderUniformInputCache {
        std::vector<uint8_t> key;
        uint64_t revision = 1;
    };
    std::unordered_map<int, ShaderUniformInputCache> shaderUniformInputCache_;
    struct ShaderInstanceCache {
        int program = -1, geometry = -1;
        std::shared_ptr<PrebuiltEntityGeometry> resource;
        std::shared_ptr<EntityBuildInput> buildInput;
        std::vector<uint8_t> key;
        uint64_t uniformRevision = 0;
        uint64_t instanceInputRevision = 0;
        VertexShaderEvaluator::Result symbolic;
        std::vector<std::array<float, 16>> vertices;
        uint64_t frame = 0;
    };
    std::unordered_map<int64_t, ShaderInstanceCache> shaderInstanceCache_;
    struct ShaderInstanceOrderKey {
        int program = -1, geometry = -1;
        std::vector<uint8_t> data, priority, input;
        uint64_t inputRevision = 0;
        bool inputDirty = true;
    };
    std::vector<ShaderInstanceOrderKey> shaderInstanceOrderKeys_;
    std::vector<size_t> shaderInstanceOrder_;
    uint64_t shaderInstanceInputRevision_ = 0;

    std::vector<std::shared_ptr<vk::BLASBatchBuilder>> pendingPrebuiltBLAS_;
    std::shared_ptr<EntityBatch> entityBatch_;
    std::shared_ptr<EntityPostBatch> entityPostBatch_;
    std::shared_ptr<EntityPostBuildBatch> entityPostBuildBatch_;
    std::vector<std::shared_ptr<EntityBuildInput>> queuedEntityBuildInputs_;
    std::unordered_map<int64_t, std::shared_ptr<Entity>> cachedEntitiesByKey_;
    uint64_t currentFrameIndex_ = 0;
    int64_t nextTransientEntityKey_ = std::numeric_limits<int64_t>::min();

    struct FrameCapacityCache {
        uint32_t slotCapacity = 512;
        uint32_t usedSlots = 0;
        uint32_t lowWaterFrames = 0;
        uint32_t pbrVertexCapacity = 0;
        uint32_t indexCapacity = 0;
        uint32_t entityLightCapacity = 0;
        VkDeviceSize blasStorageCapacity = 0;
        VkDeviceSize blasScratchCapacity = 0;
        std::shared_ptr<vk::DeviceLocalBuffer> pbrBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> entityLightBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> blasStorageBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> blasScratchBuffer;
        std::vector<std::shared_ptr<vk::BLAS>> cachedBlases;
        std::shared_ptr<vk::BLASBatchBuilder> blasBuildBatch;
        std::vector<std::vector<uint32_t>> blasGeometryKeys;
        std::vector<VkDeviceSize> blasOffsets;
        std::vector<VkDeviceSize> blasSizes;
        uint32_t ownerContext = std::numeric_limits<uint32_t>::max();
    };
    std::vector<FrameCapacityCache> frameCapacityCaches_;
    uint32_t lastFrameCacheSlot_ = std::numeric_limits<uint32_t>::max();

    std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder_;
};

struct PrebuiltEntityGeometry : public SharedObject<PrebuiltEntityGeometry> {
    std::shared_ptr<Entity> geometry;
    std::shared_ptr<EntityBuildInput> input;
    uint64_t lightRevision = std::numeric_limits<uint64_t>::max();
    Entities::VertexShaderEvaluator::Vertex programBounds;
    std::shared_ptr<vk::DeviceLocalBuffer> programBuffer;

    bool mayHaveEmission(const Emission &emission, const std::array<float, 260> &attributes);
    uint64_t emissionBoundsRevision = std::numeric_limits<uint64_t>::max();
    std::array<float, 40> emissionUvMapping{};
    bool emissionBoundsOccupied = true;
};

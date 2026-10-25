#pragma once

#include "common/shared.hpp"
#include "core/render/modules/world/dlss/dlss_wrapper.hpp"
#include "core/render/modules/world/world_module.hpp"

#include <array>
#include <chrono>

class Framework;
struct FrameworkContext;
class WorldPipeline;
struct WorldPipelineContext;

class DlssSRModuleContext;

class DlssSRModule : public WorldModule, public SharedObject<DlssSRModule> {
    friend class DlssSRModuleContext;

  public:
    enum class QualityMode {
        Quality,
        Balanced,
        Performance,
        UltraPerformance,
        DLAA,
    };

    static constexpr std::string_view NAME = "render_pipeline.module.dlss_sr.name";
    static constexpr uint32_t inputImageNum = 5;
    static constexpr uint32_t outputImageNum = 4;

    static bool initNGXContext();
    static void deinitNGXContext();
    static bool parseQualityModeValue(const std::string &value, QualityMode &outMode);

    DlssSRModule() = default;

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;
    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;
    void build() override;
    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;
    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;
    void preClose() override;

  private:
    static NVSDK_NGX_PerfQuality_Value toNgxQualityMode(QualityMode mode);
    bool resolveInputSize();
    void initDescriptorTables();
    void initPreparedImages();
    void initPipelines();

    static std::shared_ptr<NgxContext> ngxContext_;

    QualityMode qualityMode_ = QualityMode::Balanced;
    NVSDK_NGX_PerfQuality_Value ngxQualityMode_ = NVSDK_NGX_PerfQuality_Value_Balanced;
    NVSDK_NGX_DLSS_Hint_Render_Preset preset_ = NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
    NgxContext::SupportedSizes supportedSizes_{};

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t displayWidth_ = 0;
    uint32_t displayHeight_ = 0;

    std::shared_ptr<DlssSR> dlss_;
    bool initialized_ = false;

    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, inputImageNum>> inputImages_;
    std::vector<std::array<std::shared_ptr<vk::DeviceLocalImage>, outputImageNum>> outputImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> deviceDepthImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> dlssMotionVectorImages_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> descriptorTables_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> guideDescriptorTables_;
    std::shared_ptr<vk::ComputePipeline> depthPreparationPipeline_;
    std::shared_ptr<vk::ComputePipeline> firstHitDepthUpscalePipeline_;
    std::shared_ptr<vk::ComputePipeline> motionUpscalePipeline_;
    std::shared_ptr<vk::ComputePipeline> normalRoughnessUpscalePipeline_;

    glm::vec3 lastCameraPos_ = glm::vec3(0.0f);
    glm::vec3 lastCameraDir_ = glm::vec3(0.0f, 0.0f, -1.0f);
    bool firstFrame_ = true;
    std::chrono::steady_clock::time_point lastFrameTime_{};

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

class DlssSRModuleContext : public WorldModuleContext, public SharedObject<DlssSRModuleContext> {
  public:
    DlssSRModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                        std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                        std::shared_ptr<DlssSRModule> module);

    void render() override;
    bool populateFrameGenerationInputs(mcvr::FrameGenerationInputs &inputs) override;

  private:
    bool checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir);

    std::weak_ptr<DlssSRModule> module_;
    std::shared_ptr<vk::DeviceLocalImage> inputColorImage_;
    std::shared_ptr<vk::DeviceLocalImage> inputLinearDepthImage_;
    std::shared_ptr<vk::DeviceLocalImage> inputMotionVectorImage_;
    std::shared_ptr<vk::DeviceLocalImage> inputFirstHitDepthImage_;
    std::shared_ptr<vk::DeviceLocalImage> inputNormalRoughnessImage_;
    std::shared_ptr<vk::DeviceLocalImage> outputImage_;
    std::shared_ptr<vk::DeviceLocalImage> upscaledFirstHitDepthImage_;
    std::shared_ptr<vk::DeviceLocalImage> upscaledMotionVectorImage_;
    std::shared_ptr<vk::DeviceLocalImage> upscaledNormalRoughnessImage_;
    std::shared_ptr<vk::DeviceLocalImage> deviceDepthImage_;
    std::shared_ptr<vk::DeviceLocalImage> dlssMotionVectorImage_;
    std::shared_ptr<vk::DescriptorTable> descriptorTable_;
    std::shared_ptr<vk::DescriptorTable> guideDescriptorTable_;
};

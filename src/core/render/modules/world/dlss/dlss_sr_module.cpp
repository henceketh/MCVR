#include "core/render/modules/world/dlss/dlss_sr_module.hpp"

#include "core/profiler/profiler.hpp"
#include "core/render/buffers.hpp"
#include "core/render/frame_generation.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

struct DepthPreparationPushConstants {
    float cameraNear;
    float cameraFar;
    uint32_t width;
    uint32_t height;
    float jitterX;
    float jitterY;
};

static_assert(sizeof(DepthPreparationPushConstants) <= sizeof(MotionUpscalePushConstants));

constexpr VkPipelineStageFlags2 kRenderWriteStages = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT;
constexpr VkAccessFlags2 kRenderReadWriteAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

}

std::shared_ptr<NgxContext> DlssSRModule::ngxContext_ = nullptr;

bool DlssSRModule::initNGXContext() {
    std::filesystem::path dlssPath = Renderer::folderPath / "dlss";
    std::error_code ec;
    if (!std::filesystem::create_directories(dlssPath, ec) && ec) {
        std::cerr << "Failed to create DLSS directory: " << ec.message() << std::endl;
        return false;
    }

    auto framework = Renderer::instance().framework();
    if (!framework) { return false; }

    NgxContext::NgxInitInfo ngxInitInfo{};
    ngxInitInfo.instance = framework->instance();
    ngxInitInfo.physicalDevice = framework->physicalDevice();
    ngxInitInfo.device = framework->device();
    ngxInitInfo.applicationPath = dlssPath.string();

    ngxContext_ = NgxContext::acquireShared(ngxInitInfo);
    if (!ngxContext_) { return false; }
    if (NVSDK_NGX_FAILED(ngxContext_->queryDlssSRAvailable())) {
        ngxContext_.reset();
        return false;
    }

    return true;
}

void DlssSRModule::deinitNGXContext() {
    ngxContext_.reset();
}

bool DlssSRModule::parseQualityModeValue(const std::string &value, QualityMode &outMode) {
    if (value == "render_pipeline.module.dlss_sr.attribute.quality_mode.quality") {
        outMode = QualityMode::Quality;
        return true;
    }
    if (value == "render_pipeline.module.dlss_sr.attribute.quality_mode.balanced") {
        outMode = QualityMode::Balanced;
        return true;
    }
    if (value == "render_pipeline.module.dlss_sr.attribute.quality_mode.performance") {
        outMode = QualityMode::Performance;
        return true;
    }
    if (value == "render_pipeline.module.dlss_sr.attribute.quality_mode.ultra_performance") {
        outMode = QualityMode::UltraPerformance;
        return true;
    }
    if (value == "render_pipeline.module.dlss_sr.attribute.quality_mode.dlaa") {
        outMode = QualityMode::DLAA;
        return true;
    }
    return false;
}

NVSDK_NGX_PerfQuality_Value DlssSRModule::toNgxQualityMode(QualityMode mode) {
    switch (mode) {
        case QualityMode::Quality: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case QualityMode::Balanced: return NVSDK_NGX_PerfQuality_Value_Balanced;
        case QualityMode::Performance: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case QualityMode::UltraPerformance: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
        case QualityMode::DLAA: return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    return NVSDK_NGX_PerfQuality_Value_MaxQuality;
}

void DlssSRModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    const uint32_t imageCount = framework->swapchain()->imageCount();
    inputImages_.resize(imageCount);
    outputImages_.resize(imageCount);
    deviceDepthImages_.resize(imageCount);
    dlssMotionVectorImages_.resize(imageCount);
    descriptorTables_.resize(imageCount);
    guideDescriptorTables_.resize(imageCount);
    dlss_ = DlssSR::create();
}

bool DlssSRModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                           std::vector<VkFormat> &formats,
                                           uint32_t frameIndex) {
    if (images.size() != outputImageNum || formats.size() != outputImageNum) { return false; }

    auto framework = framework_.lock();
    if (!framework) { return false; }

    if (displayWidth_ == 0 || displayHeight_ == 0) {
        for (const auto &image : images) {
            if (!image) { continue; }
            displayWidth_ = image->width();
            displayHeight_ = image->height();
            break;
        }
        if (displayWidth_ == 0 || displayHeight_ == 0) {
            const VkExtent2D extent = framework->swapchain()->vkExtent();
            displayWidth_ = extent.width;
            displayHeight_ = extent.height;
        }
    }

    for (uint32_t i = 0; i < outputImageNum; ++i) {
        if (!images[i]) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, displayWidth_, displayHeight_, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        } else if (images[i]->width() != displayWidth_ || images[i]->height() != displayHeight_) {
            return false;
        }
        outputImages_[frameIndex][i] = images[i];
    }

    return true;
}

bool DlssSRModule::resolveInputSize() {
    if (!ngxContext_ || displayWidth_ == 0 || displayHeight_ == 0) { return false; }

    ngxQualityMode_ = toNgxQualityMode(qualityMode_);
    NgxContext::QuerySizeInfo queryInfo{};
    queryInfo.outputSize = {displayWidth_, displayHeight_};
    queryInfo.quality = ngxQualityMode_;
    if (NVSDK_NGX_FAILED(ngxContext_->querySupportedDlssSRInputSizes(queryInfo, supportedSizes_))) { return false; }

    const VkExtent2D inputSize =
        qualityMode_ == QualityMode::DLAA ? VkExtent2D{displayWidth_, displayHeight_} : supportedSizes_.optimalSize;
    if (inputSize.width == 0 || inputSize.height == 0) { return false; }

    renderWidth_ = inputSize.width;
    renderHeight_ = inputSize.height;
    return true;
}

bool DlssSRModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                          std::vector<VkFormat> &formats,
                                          uint32_t frameIndex) {
    if (images.size() != inputImageNum || formats.size() != inputImageNum || !ngxContext_) { return false; }
    auto framework = framework_.lock();
    if (!framework) { return false; }

    if ((renderWidth_ == 0 || renderHeight_ == 0) && !resolveInputSize()) { return false; }

    for (uint32_t i = 0; i < inputImageNum; ++i) {
        if (!images[i]) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, renderWidth_, renderHeight_, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        } else if (images[i]->width() != renderWidth_ || images[i]->height() != renderHeight_) {
            return false;
        }
        inputImages_[frameIndex][i] = images[i];
    }

    return true;
}

void DlssSRModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; ++i) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];
        if (key == "render_pipeline.module.dlss_sr.attribute.model") {
            if (value == "render_pipeline.module.dlss_sr.attribute.model.j") {
                preset_ = NVSDK_NGX_DLSS_Hint_Render_Preset_J;
            } else if (value == "render_pipeline.module.dlss_sr.attribute.model.k") {
                preset_ = NVSDK_NGX_DLSS_Hint_Render_Preset_K;
            } else if (value == "render_pipeline.module.dlss_sr.attribute.model.l") {
                preset_ = NVSDK_NGX_DLSS_Hint_Render_Preset_L;
            } else if (value == "render_pipeline.module.dlss_sr.attribute.model.m") {
                preset_ = NVSDK_NGX_DLSS_Hint_Render_Preset_M;
            } else {
                preset_ = NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
            }
            continue;
        }
        if (key != "render_pipeline.module.dlss_sr.attribute.quality_mode") { continue; }

        QualityMode parsedMode = qualityMode_;
        if (parseQualityModeValue(value, parsedMode)) {
            qualityMode_ = parsedMode;
            renderWidth_ = 0;
            renderHeight_ = 0;
        }
    }
}

void DlssSRModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    if (!framework || !worldPipeline || !ngxContext_ || !dlss_) {
        throw std::runtime_error("DLSS SR module cannot initialize without its shared NGX context");
    }
    if (renderWidth_ == 0 || renderHeight_ == 0 || displayWidth_ == 0 || displayHeight_ == 0) {
        throw std::runtime_error("DLSS SR module has unresolved render or display dimensions");
    }

    NgxContext::DlssSRInitInfo initInfo{};
    initInfo.inputSize = {renderWidth_, renderHeight_};
    initInfo.outputSize = {displayWidth_, displayHeight_};
    initInfo.quality = ngxQualityMode_;
    initInfo.preset = preset_;
    initInfo.featureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

    NVSDK_NGX_Result result = ngxContext_->initDlssSR(initInfo, framework->mainCommandPool(), dlss_);
    if (NVSDK_NGX_FAILED(result)) {
        throw std::runtime_error("DLSS SR feature creation failed: " + getNGXResultString(result));
    }
    initialized_ = true;
    firstFrame_ = true;
    lastFrameTime_ = {};

    initDescriptorTables();
    initPreparedImages();
    initPipelines();

    const uint32_t imageCount = framework->swapchain()->imageCount();
    contexts_.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; ++i) {
        contexts_[i] =
            DlssSRModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

void DlssSRModule::initDescriptorTables() {
    auto framework = framework_.lock();
    const uint32_t imageCount = framework->swapchain()->imageCount();
    auto createTable = [&]() {
        vk::DescriptorTableBuilder builder;
        auto &bindings = builder.beginDescriptorLayoutSet().beginDescriptorLayoutSetBinding();
        for (uint32_t binding = 0; binding < 7; ++binding) {
            bindings.defineDescriptorLayoutSetBinding({
                .binding = binding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            });
        }
        bindings.endDescriptorLayoutSetBinding().endDescriptorLayoutSet().definePushConstant({
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(MotionUpscalePushConstants),
        });
        return builder.build(framework->device());
    };
    for (uint32_t i = 0; i < imageCount; ++i) {
        descriptorTables_[i] = createTable();
        guideDescriptorTables_[i] = createTable();
    }
}

void DlssSRModule::initPreparedImages() {
    auto framework = framework_.lock();
    const uint32_t imageCount = framework->swapchain()->imageCount();
    for (uint32_t i = 0; i < imageCount; ++i) {
        deviceDepthImages_[i] =
            vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, renderWidth_, renderHeight_, 1,
                                         VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        dlssMotionVectorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, renderWidth_, renderHeight_, 1, VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        descriptorTables_[i]->bindImage(inputImages_[i][1], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        descriptorTables_[i]->bindImage(deviceDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        descriptorTables_[i]->bindImage(inputImages_[i][2], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        descriptorTables_[i]->bindImage(dlssMotionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);
        guideDescriptorTables_[i]->bindImage(inputImages_[i][3], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        guideDescriptorTables_[i]->bindImage(outputImages_[i][1], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        guideDescriptorTables_[i]->bindImage(inputImages_[i][2], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        guideDescriptorTables_[i]->bindImage(outputImages_[i][2], VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        guideDescriptorTables_[i]->bindImage(inputImages_[i][4], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        guideDescriptorTables_[i]->bindImage(outputImages_[i][3], VK_IMAGE_LAYOUT_GENERAL, 0, 6);
    }
}

void DlssSRModule::initPipelines() {
    auto framework = framework_.lock();
    auto depthPreparationShader =
        vk::Shader::create(framework->device(),
                           (Renderer::folderPath / "shaders/world/upscaler/linear_to_device_depth_comp.spv").string());
    auto firstHitDepthShader =
        vk::Shader::create(framework->device(),
                           (Renderer::folderPath / "shaders/world/upscaler/upscale_first_hit_depth_comp.spv").string());
    auto motionShader = vk::Shader::create(
        framework->device(), (Renderer::folderPath / "shaders/world/upscaler/upscale_motion_vector_comp.spv").string());
    auto normalRoughnessShader = vk::Shader::create(
        framework->device(),
        (Renderer::folderPath / "shaders/world/upscaler/upscale_normal_roughness_comp.spv").string());

    depthPreparationPipeline_ = vk::ComputePipelineBuilder{}
                                    .defineShader(depthPreparationShader)
                                    .definePipelineLayout(descriptorTables_[0])
                                    .build(framework->device());
    firstHitDepthUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                        .defineShader(firstHitDepthShader)
                                        .definePipelineLayout(guideDescriptorTables_[0])
                                        .build(framework->device());
    motionUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                 .defineShader(motionShader)
                                 .definePipelineLayout(guideDescriptorTables_[0])
                                 .build(framework->device());
    normalRoughnessUpscalePipeline_ = vk::ComputePipelineBuilder{}
                                          .defineShader(normalRoughnessShader)
                                          .definePipelineLayout(guideDescriptorTables_[0])
                                          .build(framework->device());
}

std::vector<std::shared_ptr<WorldModuleContext>> &DlssSRModule::contexts() {
    return contexts_;
}

void DlssSRModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                               std::shared_ptr<vk::DeviceLocalImage> image,
                               int index) {}

void DlssSRModule::preClose() {
    if (auto framework = framework_.lock()) { framework->waitRenderQueueIdle(); }
    if (dlss_) { dlss_->deinit(); }
    initialized_ = false;
}

DlssSRModuleContext::DlssSRModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                         std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                         std::shared_ptr<DlssSRModule> module)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      module_(module),
      inputColorImage_(module->inputImages_[frameworkContext->frameIndex][0]),
      inputLinearDepthImage_(module->inputImages_[frameworkContext->frameIndex][1]),
      inputMotionVectorImage_(module->inputImages_[frameworkContext->frameIndex][2]),
      inputFirstHitDepthImage_(module->inputImages_[frameworkContext->frameIndex][3]),
      inputNormalRoughnessImage_(module->inputImages_[frameworkContext->frameIndex][4]),
      outputImage_(module->outputImages_[frameworkContext->frameIndex][0]),
      upscaledFirstHitDepthImage_(module->outputImages_[frameworkContext->frameIndex][1]),
      upscaledMotionVectorImage_(module->outputImages_[frameworkContext->frameIndex][2]),
      upscaledNormalRoughnessImage_(module->outputImages_[frameworkContext->frameIndex][3]),
      deviceDepthImage_(module->deviceDepthImages_[frameworkContext->frameIndex]),
      dlssMotionVectorImage_(module->dlssMotionVectorImages_[frameworkContext->frameIndex]),
      descriptorTable_(module->descriptorTables_[frameworkContext->frameIndex]),
      guideDescriptorTable_(module->guideDescriptorTables_[frameworkContext->frameIndex]) {}

bool DlssSRModuleContext::populateFrameGenerationInputs(mcvr::FrameGenerationInputs &inputs) {
    auto module = module_.lock();
    if (!module || !module->initialized_ || !deviceDepthImage_ || !dlssMotionVectorImage_) { return false; }

    inputs.depth = deviceDepthImage_;
    inputs.motionVector = dlssMotionVectorImage_;
    inputs.renderWidth = module->renderWidth_;
    inputs.renderHeight = module->renderHeight_;
    inputs.motionVectorScale = glm::vec2(1.0f);
    inputs.depthInverted = false;
    inputs.cameraMotionIncluded = true;
    return true;
}

bool DlssSRModuleContext::checkCameraReset(const glm::vec3 &cameraPos, const glm::vec3 &cameraDir) {
    auto module = module_.lock();
    if (!module) { return true; }

    if (module->firstFrame_ || !Renderer::instance().buffers()->hasWorldHistory()) {
        module->firstFrame_ = false;
        module->lastFrameTime_ = {};
        module->lastCameraPos_ = cameraPos;
        module->lastCameraDir_ = cameraDir;
        return true;
    }

    const float positionDelta = glm::length(cameraPos - module->lastCameraPos_);
    const float currentDirectionLength = glm::length(cameraDir);
    const float previousDirectionLength = glm::length(module->lastCameraDir_);
    const float directionDot =
        currentDirectionLength > 1e-6f && previousDirectionLength > 1e-6f ?
            glm::dot(cameraDir / currentDirectionLength, module->lastCameraDir_ / previousDirectionLength) :
            1.0f;
    const bool shouldReset = positionDelta > 1.0f || directionDot < 0.866f;

    module->lastCameraPos_ = cameraPos;
    module->lastCameraDir_ = cameraDir;
    return shouldReset;
}

void DlssSRModuleContext::render() {
    MCVR_PROFILE_SCOPE("native.DlssSRModuleContext.render");
    auto module = module_.lock();
    auto context = frameworkContext.lock();
    if (!module || !context || !module->initialized_ || !module->dlss_ || !inputColorImage_ ||
        !inputLinearDepthImage_ || !inputMotionVectorImage_ || !inputFirstHitDepthImage_ ||
        !inputNormalRoughnessImage_ || !outputImage_ || !upscaledFirstHitDepthImage_ || !upscaledMotionVectorImage_ ||
        !upscaledNormalRoughnessImage_ || !deviceDepthImage_ || !dlssMotionVectorImage_ || !descriptorTable_ ||
        !guideDescriptorTable_) {
        return;
    }

    auto framework = context->framework.lock();
    if (!framework) { return; }
    auto worldCommandBuffer = context->worldCommandBuffer;
    if (!worldCommandBuffer) { return; }
    const uint32_t queueIndex = framework->physicalDevice()->mainQueueIndex();

    auto transition = [&](const std::shared_ptr<vk::DeviceLocalImage> &image, VkPipelineStageFlags2 srcStage,
                          VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageLayout newLayout) {
        worldCommandBuffer->barriersBufferImage({}, {{.srcStageMask = srcStage,
                                                      .srcAccessMask = srcAccess,
                                                      .dstStageMask = dstStage,
                                                      .dstAccessMask = dstAccess,
                                                      .oldLayout = image->imageLayout(),
                                                      .newLayout = newLayout,
                                                      .srcQueueFamilyIndex = queueIndex,
                                                      .dstQueueFamilyIndex = queueIndex,
                                                      .image = image,
                                                      .subresourceRange = vk::wholeColorSubresourceRange}});
        image->imageLayout() = newLayout;
    };

    auto worldUBO =
        static_cast<vk::Data::WorldUBO *>(Renderer::instance().buffers()->worldUniformBuffer()->mappedPtr());
    if (!worldUBO) { return; }

    auto dispatchAuxiliaryUpscale = [&](const std::shared_ptr<vk::DeviceLocalImage> &input,
                                        const std::shared_ptr<vk::DeviceLocalImage> &output,
                                        const std::shared_ptr<vk::ComputePipeline> &pipeline) {
        transition(input, kRenderWriteStages, kRenderReadWriteAccess, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
        transition(output, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   kRenderReadWriteAccess, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_GENERAL);

        const MotionUpscalePushConstants pushConstants{
            static_cast<float>(module->displayWidth_) / static_cast<float>(std::max(1u, module->renderWidth_)),
            static_cast<float>(module->displayHeight_) / static_cast<float>(std::max(1u, module->renderHeight_)),
            module->renderWidth_,
            module->renderHeight_,
            module->displayWidth_,
            module->displayHeight_,
            worldUBO->cameraJitter.x,
            worldUBO->cameraJitter.y,
        };
        worldCommandBuffer->bindDescriptorTable(guideDescriptorTable_, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(pipeline);
        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), guideDescriptorTable_->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->displayWidth_ + 15) / 16,
                      (module->displayHeight_ + 15) / 16, 1);
    };

    transition(inputLinearDepthImage_, kRenderWriteStages, kRenderReadWriteAccess,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
    transition(inputMotionVectorImage_, kRenderWriteStages, kRenderReadWriteAccess,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
    transition(deviceDepthImage_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kRenderReadWriteAccess,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);
    transition(dlssMotionVectorImage_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kRenderReadWriteAccess,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);

    const DepthPreparationPushConstants depthPushConstants{
        0.1f, 10000.0f, module->renderWidth_, module->renderHeight_, worldUBO->cameraJitter.x, worldUBO->cameraJitter.y,
    };
    worldCommandBuffer->bindDescriptorTable(descriptorTable_, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->depthPreparationPipeline_);
    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), descriptorTable_->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(depthPushConstants), &depthPushConstants);
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->renderWidth_ + 15) / 16,
                  (module->renderHeight_ + 15) / 16, 1);

    transition(inputColorImage_, kRenderWriteStages, kRenderReadWriteAccess, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition(deviceDepthImage_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition(dlssMotionVectorImage_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    transition(outputImage_, kRenderWriteStages, kRenderReadWriteAccess,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
               VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_IMAGE_LAYOUT_GENERAL);

    module->dlss_->setResource(DlssSR::RESOURCE_COLOR_IN, inputColorImage_);
    module->dlss_->setResource(DlssSR::RESOURCE_COLOR_OUT, outputImage_);
    module->dlss_->setResource(DlssSR::RESOURCE_MOTIONVECTOR, dlssMotionVectorImage_);
    module->dlss_->setResource(DlssSR::RESOURCE_DEPTH, deviceDepthImage_);

    const glm::vec3 cameraDirection(worldUBO->cameraViewMat[0][2], worldUBO->cameraViewMat[1][2],
                                    worldUBO->cameraViewMat[2][2]);
    const bool reset = checkCameraReset(glm::vec3(worldUBO->cameraPos), cameraDirection);
    const auto now = std::chrono::steady_clock::now();
    const float frameTime = module->lastFrameTime_ == std::chrono::steady_clock::time_point{} ? 1000.0f / 60.0f :
        std::chrono::duration<float, std::milli>(now - module->lastFrameTime_).count();
    module->lastFrameTime_ = now;
    const NVSDK_NGX_Result evaluateResult =
        module->dlss_->evaluate(worldCommandBuffer, {module->renderWidth_, module->renderHeight_},
                                worldUBO->cameraJitter, glm::vec2(1.0f), reset, frameTime);
    if (NVSDK_NGX_FAILED(evaluateResult)) { return; }

    inputColorImage_->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    deviceDepthImage_->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dlssMotionVectorImage_->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    outputImage_->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    dispatchAuxiliaryUpscale(inputFirstHitDepthImage_, upscaledFirstHitDepthImage_,
                             module->firstHitDepthUpscalePipeline_);
    dispatchAuxiliaryUpscale(inputMotionVectorImage_, upscaledMotionVectorImage_, module->motionUpscalePipeline_);
    dispatchAuxiliaryUpscale(inputNormalRoughnessImage_, upscaledNormalRoughnessImage_,
                             module->normalRoughnessUpscalePipeline_);
}

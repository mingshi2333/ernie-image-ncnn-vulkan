// SPDX-License-Identifier: MIT
#pragma once
#include "mat.h"
#include "option.h"
#include <vector>
#if NCNN_VULKAN
#include "gpu.h"
#include "command.h"
#include "pipeline.h"
#endif

namespace ernie {
// Batch-one, deterministic FlowMatch Euler with the official Turbo shift=4.
// Arithmetic and latent storage in these initial components are FP32.
struct FlowSchedule
{
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    static FlowSchedule turbo(int steps = 8);
    float delta(size_t step) const;
};

void euler_step(const ncnn::Mat& sample, const ncnn::Mat& prediction,
                float delta, ncnn::Mat& next, int threads = 4);
bool finite_latent(const ncnn::Mat& value);
// Pipeline epsilon is 1e-5, despite the currently published VAE config.
// BN unnormalization precedes channel-to-spatial unpatchification.
ncnn::Mat unpack_for_vae(const ncnn::Mat& packed, const ncnn::Mat& mean,
                         const ncnn::Mat& variance, int threads = 4);

#if NCNN_VULKAN
// Caller owns allocators, VkMat lifetimes and VkCompute submission. This object
// and all referenced tensors must outlive the submitted command buffer.
// Inputs must be pack1 FP32. Low precision scheduler semantics are not implied.
// Validate BN statistics while loading on the host before uploading them.
class VulkanLatentOps
{
public:
    explicit VulkanLatentOps(const ncnn::VulkanDevice* device);
    ~VulkanLatentOps();
    VulkanLatentOps(const VulkanLatentOps&) = delete;
    VulkanLatentOps& operator=(const VulkanLatentOps&) = delete;
    void record_euler(const ncnn::VkMat& sample, const ncnn::VkMat& prediction,
                      float delta, ncnn::VkMat& next, ncnn::VkCompute& command,
                      const ncnn::Option& option) const;
    void record_unpack(const ncnn::VkMat& packed, const ncnn::VkMat& mean,
                       const ncnn::VkMat& variance, ncnn::VkMat& unpacked,
                       ncnn::VkCompute& command, const ncnn::Option& option) const;
    // Download 128 finite-status floats (512 bytes), never the activation.
    bool finite_latent(const ncnn::VkMat& value, const ncnn::VulkanDevice* device,
                       const ncnn::Option& option) const;
private:
    ncnn::Pipeline* euler_ = nullptr;
    ncnn::Pipeline* unpack_ = nullptr;
    ncnn::Pipeline* finite_ = nullptr;
};
#endif
} // namespace ernie

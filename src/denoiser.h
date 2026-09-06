// SPDX-License-Identifier: MIT
#pragma once
#include "dit.h"
#include "latent_ops.h"
#include <functional>

namespace ernie
{
// ERNIE Timesteps(4096, flip_sin_to_cos=false, downscale_freq_shift=0).
// The timestep and the master Euler latent are FP32 in this API. Selecting
// low-precision DiT storage does not imply the official low-dtype scheduler.
ncnn::Mat timestep_features(float timestep);
struct DenoiseModel
{
    ComponentFiles input_head, output_head;
    std::vector<ComponentFiles> blocks;
};
struct DenoiseStepStats
{
    float timestep = 0, delta = 0;
    DitStats dit;
    double elapsed_seconds = 0;
};
using CpuStepObserver = std::function<void(size_t, const ncnn::Mat &, const ncnn::Mat &)>;
// Constants: text embeddings, cosine, sine, attention mask. The optional
// observer sees the prediction and updated latent after each Euler step.
// start_step retains the original schedule and absolute observer step indices;
// stats contains only executed steps. start_step=steps performs no model work.
ncnn::Mat denoise(const DenoiseModel &model, const ncnn::Mat &initial,
                  const std::vector<ncnn::Mat> &constants, int steps, const ncnn::Option &option,
                  std::vector<DenoiseStepStats> &stats, const CpuStepObserver &observer = {}, int start_step = 0);
#if NCNN_VULKAN
using VulkanStepObserver = std::function<void(size_t, const ncnn::VkMat &, const ncnn::VkMat &)>;
// Initial sample is pack1 FP32, constants use the DiT storage options. All
// activations stay on device unless the caller installs a downloading observer.
// Session allocators and optional shared pipeline cache must outlive this call.
ncnn::VkMat denoise(const DenoiseModel &model, const ncnn::VkMat &initial,
                    const std::vector<ncnn::VkMat> &constants, int steps, const ncnn::VulkanDevice *device,
                    const ncnn::Option &option, std::vector<DenoiseStepStats> &stats,
                    const VulkanStepObserver &observer = {}, int start_step = 0);
#endif
} // namespace ernie

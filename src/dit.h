// SPDX-License-Identifier: MIT
#pragma once
#include "block_sequence.h"

namespace ernie {
struct DitStats
{
    BlockSequenceStats blocks;
    double input_head_seconds = 0;
    double output_head_seconds = 0;
};
// One prediction, with caller-supplied saved time features, RoPE and mask.
// This API does not tokenize, encode text, sample noise, run Euler, or decode VAE.
ncnn::Mat run_dit(const std::string& input_head, const std::vector<std::string>& blocks,
    const std::string& output_head, const std::vector<ncnn::Mat>& inputs,
    const ncnn::Option& option, DitStats& stats, const CpuStageObserver& observer = {});
#if NCNN_VULKAN
// Inputs: packed latent NCHW, text matrix, time features, cosine, sine, mask.
// Heads and individual blocks release weights between stages. All activation
// storage belongs to caller-owned allocators; no intermediate download.
// A session-owned Option::pipeline_cache avoids rebuilding the same Vulkan
// pipelines for each streamed Net. It must outlive every inference command.
ncnn::VkMat run_dit(const std::string& input_head, const std::vector<std::string>& blocks,
    const std::string& output_head, const std::vector<ncnn::VkMat>& inputs,
    const ncnn::VulkanDevice* device, const ncnn::Option& option, DitStats& stats,
    const VulkanStageObserver& observer = {});
#endif
}

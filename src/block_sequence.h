// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"
#include <string>
#include <vector>

namespace ernie {
enum class WeightPolicy { Stream, Resident };
struct BlockSequenceStats
{
    std::vector<double> load_seconds;
    std::vector<double> compute_seconds;
    int peak_loaded_nets = 0;
    int compute_submissions = 0;
};

// Every model must be a verified static graph for the same token bucket. The
// nine constants are shared AdaLN (six tensors), cos, sin, and attention mask.
// Stream destroys each block's Net after completion and retains its output in
// session-owned storage. Resident is a bounded comparison, not an 8GB strategy.
ncnn::Mat run_block_sequence(const std::vector<std::string>& models, const ncnn::Mat& input,
    const std::vector<ncnn::Mat>& constants, const ncnn::Option& option,
    WeightPolicy policy, BlockSequenceStats& stats);

#if NCNN_VULKAN
// No intermediate activation download. Caller keeps all allocator owners alive
// until returned VkMat and inputs are released. Each block has one submission,
// allowing its weights to be released after the GPU has finished consuming them.
ncnn::VkMat run_block_sequence(const std::vector<std::string>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats);
#endif
} // namespace ernie

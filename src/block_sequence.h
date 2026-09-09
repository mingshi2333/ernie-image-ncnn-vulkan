// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "weight_placement.h"
#include "weight_session.h"
#include "memory_execution.h"
#include "net.h"
#include <functional>
#include <string>
#include <vector>

namespace ernie {
// Optional diagnostics. Observers run after completed stages and must not
// mutate or retain the session-owned tensor. They are absent in normal runs.
using CpuStageObserver = std::function<void(const std::string&, const ncnn::Mat&)>;
#if NCNN_VULKAN
using VulkanStageObserver = std::function<void(const std::string&, const ncnn::VkMat&)>;
#endif
enum class WeightPolicy { Stream, Resident };
struct BlockSequenceStats
{
    struct Detail { int block=-1; std::string boundary,status; double seconds=0; };
    std::vector<double> load_seconds;
    std::vector<double> compute_seconds;
    int peak_loaded_nets = 0;
    int compute_submissions = 0;
    bool collect_details = false;
    std::vector<Detail> details;
    std::uint64_t prefetch_started = 0, prefetch_used = 0, prefetch_skipped = 0;
    std::uint64_t prefetch_peak_charged_bytes = 0;
    double prefetch_overlap_seconds = 0;
};

// Every model must be a verified static graph for the same token bucket. The
// nine constants are shared AdaLN (six tensors), cos, sin, and attention mask.
// Stream destroys each block's Net after completion and retains its output in
// session-owned storage. Resident is a bounded comparison, not an 8GB strategy.
ncnn::Mat run_block_sequence(const std::vector<ComponentFiles>& models, const ncnn::Mat& input,
    const std::vector<ncnn::Mat>& constants, const ncnn::Option& option,
    WeightPolicy policy, BlockSequenceStats& stats, const CpuStageObserver& observer = {});

ncnn::Mat run_block_sequence(const std::vector<std::string>& models, const ncnn::Mat& input,
    const std::vector<ncnn::Mat>& constants, const ncnn::Option& option,
    WeightPolicy policy, BlockSequenceStats& stats, const CpuStageObserver& observer = {});

#if NCNN_VULKAN
// No intermediate activation download. Caller keeps all allocator owners alive
// until returned VkMat and inputs are released. Each block completes before its
// weights are released. FP32 query chunks add submissions recorded in stats.
ncnn::VkMat run_block_sequence(const std::vector<ComponentFiles>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats,
    const VulkanStageObserver& observer = {}, WeightPlacement* placement = nullptr, WeightSession* session = nullptr,
    const MemoryExecution* memory = nullptr);

ncnn::VkMat run_block_sequence(const std::vector<std::string>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats,
    const VulkanStageObserver& observer = {}, WeightPlacement* placement = nullptr, WeightSession* session = nullptr,
    const MemoryExecution* memory = nullptr);
#endif
} // namespace ernie

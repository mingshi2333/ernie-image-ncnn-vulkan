// SPDX-License-Identifier: MIT
#pragma once
#include "denoiser.h"
#include "vulkan_memory.h"

namespace ernie {
#if NCNN_VULKAN
struct VulkanDenoiseSettings
{
    ActivationMemory memory = ActivationMemory::Auto;
    WeightMemory weights = WeightMemory::Auto;
    std::uint64_t gpu_reserve_bytes = 512ull * 1024 * 1024;
    std::uint64_t spill_bytes = 2048ull * 1024 * 1024;
    std::uint64_t cache_bytes = 0, prefetch_bytes = 0;
    std::uint64_t ram_reserve_bytes = 3072ull * 1024 * 1024;
    unsigned retries = 3;
    bool collect_details = false, download_predictions = false;
    HostAvailableReader available = host_memory_available_reader();
    // Internal test seam, absent in public generation requests. Invoked before
    // each attempted step, so bounded injected failures exercise real restart.
    std::function<void(unsigned attempt, int step)> before_step;
};
struct VulkanDenoiseStats
{
    VulkanMemoryStats memory;
    WeightSessionStats cache;
    std::uint64_t host_weight_requests = 0, device_weight_requests = 0, unavailable_queries = 0;
    std::uint64_t prefetch_started = 0, prefetch_used = 0, prefetch_skipped = 0, prefetch_peak_charged_bytes = 0;
    double prefetch_overlap_seconds = 0;
    unsigned retries = 0;
    int query_rows = 128;
};
struct VulkanDenoiseObservers
{
    WeightPlacement::Observer placement;
    std::function<void(size_t, const ncnn::Mat&, const ncnn::Mat&, const DenoiseStepStats&)> step;
    std::function<void(size_t, const DenoiseStepStats&)> failed_step;
    std::function<void(unsigned, int, int, const std::string&)> retry;
    std::function<void(bool upload, double seconds)> transfer;
};
// CPU checkpoints commit only after a complete Euler step and checked download.
// Each attempt owns its GPU allocations and weight cache. A retry reconstructs
// them from the last checkpoint with the same absolute schedule and precision.
ncnn::Mat denoise_with_recovery(const DenoiseModel&, const ncnn::Mat& initial,
    const std::vector<ncnn::Mat>& constants, int steps, int start_step,
    const ncnn::VulkanDevice*, const ncnn::Option&,
    const VulkanDenoiseSettings&, VulkanDenoiseStats&, const VulkanDenoiseObservers& = {});
#endif
} // namespace ernie

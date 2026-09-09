// SPDX-License-Identifier: MIT
#include "vulkan_denoise.h"
#if NCNN_VULKAN
#include "pipelinecache.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

namespace ernie {
namespace {
using Clock = std::chrono::steady_clock;
void add_memory(VulkanMemoryStats& total, const VulkanMemoryStats& value)
{
    total.device_allocations += value.device_allocations;
    total.host_allocations += value.host_allocations;
    total.fallbacks += value.fallbacks;
    total.allocation_failures += value.allocation_failures;
    total.host_peak_bytes = std::max(total.host_peak_bytes, value.host_peak_bytes);
    total.host_device_local_allocations += value.host_device_local_allocations;
    total.host_non_device_local_allocations += value.host_non_device_local_allocations;
}
void add_cache(WeightSessionStats& total, const WeightSessionStats& value)
{
    total.hits += value.hits; total.loads += value.loads;
    total.admissions += value.admissions; total.evictions += value.evictions;
    total.peak_bytes = std::max(total.peak_bytes, value.peak_bytes);
    total.peak_nets = std::max(total.peak_nets, value.peak_nets);
    total.unavailable_queries += value.unavailable_queries;
}
void add_prefetch(VulkanDenoiseStats& total, const BlockSequenceStats& value)
{
    total.prefetch_started += value.prefetch_started;
    total.prefetch_used += value.prefetch_used;
    total.prefetch_skipped += value.prefetch_skipped;
    total.prefetch_peak_charged_bytes = std::max(total.prefetch_peak_charged_bytes, value.prefetch_peak_charged_bytes);
    total.prefetch_overlap_seconds += value.prefetch_overlap_seconds;
}
} // namespace
ncnn::Mat denoise_with_recovery(const DenoiseModel& model, const ncnn::Mat& initial,
    const std::vector<ncnn::Mat>& constants, int steps, int start_step,
    const ncnn::VulkanDevice* device, const ncnn::Option& base,
    const VulkanDenoiseSettings& settings, VulkanDenoiseStats& total, const VulkanDenoiseObservers& observer)
{
    if (!device || !base.use_vulkan_compute || settings.retries > 3 ||
        start_step < 0 || start_step > steps || constants.size() != 4 || !finite_latent(initial))
        throw std::invalid_argument("Invalid Vulkan recovery session");
    // Validate the unchanged schedule even when start_step == steps.
    FlowSchedule::turbo(steps);
    total = {};
    ncnn::Mat checkpoint = initial;
    int resume_step = start_step;
    for (unsigned attempt = 0; resume_step < steps; ++attempt)
    {
        const int attempt_start = resume_step;
        bool observer_failed = false;
        bool synchronization_failed = false;
        std::string failure;
        auto notify = [&](const auto& callback, auto&&... args) {
            if (!callback) return;
            try { callback(std::forward<decltype(args)>(args)...); }
            catch (...) { observer_failed = true; throw; }
        };
        total.query_rows = 128 >> attempt;
        try
        {
            ncnn::PipelineCache pipelines(device);
            const auto memory_mode = attempt && settings.memory == ActivationMemory::Auto && settings.spill_bytes ?
                ActivationMemory::Host : settings.memory;
            AdaptiveVkAllocator blobs(device, memory_mode, settings.gpu_reserve_bytes,
                settings.spill_bytes, settings.ram_reserve_bytes, settings.available);
            ncnn::VkStagingAllocator staging(device);
            ncnn::Option option = base;
            option.blob_vkallocator = option.workspace_vkallocator = &blobs;
            option.staging_vkallocator = &staging;
            option.pipeline_cache = &pipelines;
            ncnn::Option high = option;
            high.use_fp16_storage = high.use_fp16_packed = high.use_fp16_arithmetic = false;
            high.use_bf16_storage = high.use_bf16_packed = high.use_packing_layout = false;
            std::unique_ptr<WeightPlacement> placement;
            std::unique_ptr<WeightSession> weights;
            std::vector<DenoiseStepStats> step_stats;
            size_t accounted = 0;
            bool finalized = false;
            auto account = [&] {
                while (accounted < step_stats.size())
                {
                    const auto k = accounted++;
                    add_prefetch(total, step_stats[k].dit.blocks);
                    notify(observer.failed_step, size_t(attempt_start) + k, step_stats[k]);
                }
                if (finalized) return;
                finalized = true;
                add_memory(total.memory, blobs.stats());
                if (placement)
                {
                    total.host_weight_requests += placement->host_requests();
                    total.device_weight_requests += placement->device_requests();
                    total.unavailable_queries += placement->unavailable_queries();
                }
                if (weights) add_cache(total.cache, weights->stats());
            };
            try
            {
                ncnn::VkMat gpu_initial;
                std::vector<ncnn::VkMat> gpu_constants(constants.size());
                {
                    const auto begin = Clock::now();
                    ncnn::VkCompute upload(device);
                    ncnn::VkMat packed;
                    upload.record_upload(checkpoint, packed, high);
                    if (packed.empty()) throw GpuAllocationError("Upload checkpoint");
                    device->convert_packing(packed, gpu_initial, 1, 1, upload, high);
                    for (size_t k = 0; k < constants.size(); ++k)
                    {
                        upload.record_upload(constants[k], gpu_constants[k], option);
                        if (gpu_constants[k].empty()) throw GpuAllocationError("Upload conditioning");
                    }
                    if (gpu_initial.empty()) throw GpuAllocationError("Allocate master latent");
                    check_ncnn_memory(upload.submit_and_wait(), "Upload recovery session");
                    notify(observer.transfer, true, std::chrono::duration<double>(Clock::now() - begin).count());
                }
                blobs.reclaim_completed();
                // Read the compute heap using a small device allocation even
                // when activations are on RAM. A host tensor's heap is different.
                const auto reader = compute_memory_budget_reader(device);
                const auto weight_mode = attempt && settings.weights == WeightMemory::Auto ? WeightMemory::Host : settings.weights;
                placement = std::make_unique<WeightPlacement>(weight_mode, settings.gpu_reserve_bytes, reader, [&](const ComponentFiles& files, const WeightPlacementDecision& decision) {
                        notify(observer.placement, files, decision);
                    });
                if (!attempt && settings.cache_bytes)
                    weights = std::make_unique<WeightSession>(WeightBudget{settings.cache_bytes, settings.ram_reserve_bytes},
                        settings.available, dit_host_weight_inspector(device));
                MemoryExecution execution;
                execution.prefetch_bytes = attempt ? 0 : settings.prefetch_bytes;
                execution.ram_reserve_bytes = settings.ram_reserve_bytes;
                execution.available = settings.available;
                execution.query_rows = total.query_rows;
                if (settings.before_step)
                    execution.before_step = [&](int step) { settings.before_step(attempt, step); };
                denoise(model, gpu_initial, gpu_constants, steps, device, option, step_stats,
                    [&](size_t i, const ncnn::VkMat& prediction, const ncnn::VkMat& sample) {
                        ncnn::Mat saved, predicted;
                        const auto begin = Clock::now();
                        {
                            ncnn::VkCompute download(device);
                            download.record_download(sample, saved, high);
                            if (settings.download_predictions) download.record_download(prediction, predicted, high);
                            check_ncnn_memory(download.submit_and_wait(), "Save completed denoise step");
                        }
                        if (saved.empty() || (settings.download_predictions && predicted.empty()))
                            throw GpuAllocationError("Save denoise checkpoint");
                        if (!finite_latent(saved)) throw std::runtime_error("Invalid denoise checkpoint");
                        // Commit after every part of the checkpoint transfer succeeds.
                        checkpoint = saved;
                        resume_step = int(i) + 1;
                        const auto& step = step_stats.at(i - size_t(attempt_start));
                        add_prefetch(total, step.dit.blocks);
                        accounted = step_stats.size();
                        notify(observer.transfer, false, std::chrono::duration<double>(Clock::now() - begin).count());
                        notify(observer.step, i, predicted, checkpoint, step);
                        blobs.reclaim_completed();
                    }, attempt_start, settings.collect_details, placement.get(), weights.get(), &execution);
                account();
            }
            catch (...)
            {
                const auto original = std::current_exception();
                // No retry reuses a command, cache or buffer from this attempt.
                // A failed wait cannot establish safe buffer retirement, even
                // when its result is an OOM code. Never retry such a failure.
                const auto synchronized = ncnn::vkDeviceWaitIdle(device->vkdevice());
                if (synchronized != VK_SUCCESS)
                {
                    // Also keep a failed diagnostic allocation out of retry.
                    synchronization_failed = true;
                    throw std::runtime_error("Synchronize failed denoise attempt failed (Vulkan " +
                        std::to_string(synchronized) + "); cannot safely retry");
                }
                account();
                std::rethrow_exception(original);
            }
        }
        catch (const GpuAllocationError& error)
        {
            if (observer_failed || synchronization_failed) throw;
            failure = error.what();
        }
        catch (const std::bad_alloc&)
        {
            if (observer_failed || synchronization_failed) throw;
            failure = "Memory allocation failed";
        }
        if (failure.empty()) break;
        if (attempt >= settings.retries)
            throw GpuAllocationError("Denoise allocation recovery exhausted after " + std::to_string(attempt) +
                " retries at step " + std::to_string(resume_step + 1) + ": " + failure);
        ++total.retries;
        if (observer.retry) observer.retry(total.retries, resume_step, 128 >> (attempt + 1), failure);
    }
    return checkpoint;
}
} // namespace ernie
#endif

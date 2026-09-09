// SPDX-License-Identifier: MIT
#include "block_sequence.h"
#include "ernie_gelu.h"
#include "ernie_attention.h"
#include "vulkan_memory.h"
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>

namespace ernie {
namespace {
using Clock = std::chrono::steady_clock;
void check(int result, const char* action)
{
    check_ncnn_memory(result, action);
}

template<class Configure>
std::unique_ptr<ncnn::Net> load(const ComponentFiles& files, const ncnn::Option& option,
                              Configure&& configure, BlockSequenceStats& stats, int block)
{
    const auto start = Clock::now();
    auto net = std::make_unique<ncnn::Net>();
    // Inference outputs use explicit session allocators via each Extractor.
    // Do not pass session allocators as weight allocators to the Net.
    net->opt = option;
#if NCNN_VULKAN
    net->opt.blob_vkallocator = net->opt.workspace_vkallocator = net->opt.staging_vkallocator = nullptr;
#endif
    configure(*net);
    check(register_layers(*net), "register ERNIE layers");
    try { load_component_param(*net, files); }
    catch (...) { if(stats.collect_details) stats.details.push_back({block,"net_setup_param","failed",std::chrono::duration<double>(Clock::now()-start).count()}); throw; }
    if(stats.collect_details) stats.details.push_back({block,"net_setup_param","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    const auto model_start=Clock::now();
    try { load_component_model(*net, files); }
    catch (...) { if(stats.collect_details) stats.details.push_back({block,"model_load_composite","failed",std::chrono::duration<double>(Clock::now()-model_start).count()}); throw; }
    const auto model_seconds=std::chrono::duration<double>(Clock::now()-model_start).count();
    if(stats.collect_details) stats.details.push_back({block,"model_load_composite","complete",model_seconds});
    stats.load_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
    return net;
}

void check_request(size_t models, size_t constants, WeightPolicy policy)
{
    if (models < 1 || models > 36 || constants != 9)
        throw std::invalid_argument("Expected 1..36 blocks and nine shared constants");
    if (policy == WeightPolicy::Resident && models > 2)
        throw std::invalid_argument("Resident comparison is bounded to two blocks; use streaming for longer sequences");
}
} // namespace

ncnn::Mat run_block_sequence(const std::vector<ComponentFiles>& models, const ncnn::Mat& input,
    const std::vector<ncnn::Mat>& constants, const ncnn::Option& option,
    WeightPolicy policy, BlockSequenceStats& stats, const CpuStageObserver& observer)
{
    check_request(models.size(), constants.size(), policy);
    if (option.use_vulkan_compute) throw std::invalid_argument("CPU sequence requires CPU options");
    const bool collect_details=stats.collect_details;
    stats = {};
    stats.collect_details=collect_details;
    auto configure = [](ncnn::Net&) {};
    std::vector<std::unique_ptr<ncnn::Net>> resident;
    if (policy == WeightPolicy::Resident)
        for (size_t i=0;i<models.size();++i) resident.push_back(load(models[i], option, configure, stats, int(i)));
    stats.peak_loaded_nets = policy == WeightPolicy::Resident ? int(models.size()) : 1;
    ncnn::Mat current = input;
    for (size_t i = 0; i < models.size(); ++i)
    {
        auto streamed = policy == WeightPolicy::Stream ? load(models[i], option, configure, stats, int(i)) : nullptr;
        auto& net = policy == WeightPolicy::Stream ? *streamed : *resident[i];
        const auto start = Clock::now();
        {
        auto extractor = net.create_extractor();
        check(extractor.input("in0", current), "input activation");
        for (size_t k = 0; k < constants.size(); ++k)
            check(extractor.input(("in" + std::to_string(k + 1)).c_str(), constants[k]), "input constant");
        ncnn::Mat next;
        try { check(extractor.extract("out0", next), "extract block output"); }
        catch (...) { if(stats.collect_details) stats.details.push_back({int(i),"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()}); throw; }
        // Extractor's default output conversion gives pack1 FP32. Mat retains
        // storage independently of Net, so it survives streamed destruction.
        current = next;
        stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        if(stats.collect_details) stats.details.push_back({int(i),"extract_compute_composite","complete",stats.compute_seconds.back()});
        if (observer) observer("block-" + std::to_string(i), current);
        }
        if (streamed) { const auto destroy=Clock::now();streamed.reset();if(stats.collect_details) stats.details.push_back({int(i),"net_destroy","complete",std::chrono::duration<double>(Clock::now()-destroy).count()}); }
    }
    return current;
}

#if NCNN_VULKAN
namespace {
std::uint64_t prepared_estimate(const ComponentFiles& files)
{
    const auto bytes = std::filesystem::file_size(files.weight_path);
    if (bytes > (UINT64_MAX - 64ull * 1024 * 1024) / 2)
        throw std::overflow_error("Prepared weight estimate overflow");
    return bytes * 2 + 64ull * 1024 * 1024;
}
bool prefetch_headroom(const MemoryExecution& memory, std::uint64_t extra)
{
    const auto available = memory.available ? memory.available() : std::nullopt;
    return available && *available >= memory.ram_reserve_bytes &&
        *available - memory.ram_reserve_bytes >= extra;
}
struct PreparedBlock
{
    std::unique_ptr<ncnn::Net> net;
    BlockSequenceStats stats;
    Clock::time_point start, end;
    std::uint64_t charge = 0;
};
} // namespace
ncnn::VkMat run_block_sequence(const std::vector<ComponentFiles>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats,
    const VulkanStageObserver& observer, WeightPlacement* placement, WeightSession* session,
    const MemoryExecution* memory)
{
    check_request(models.size(), constants.size(), policy);
    if (!device || !option.use_vulkan_compute || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Vulkan sequence requires device and session allocators");
    if (session && policy != WeightPolicy::Stream)
        throw std::invalid_argument("Weight session requires streamed execution");
    if (memory && memory->prefetch_bytes && (policy != WeightPolicy::Stream ||
        option.use_fp16_storage || option.use_fp16_packed || option.use_bf16_storage || option.use_bf16_packed))
        throw std::invalid_argument("Prefetch requires streamed Vulkan FP32 weights");
    const bool collect_details = stats.collect_details;
    stats = {};
    stats.collect_details = collect_details;
    auto configure = [device](ncnn::Net& net) { net.set_vulkan_device(device); };
    auto load_block = [&](size_t i, bool prefer_host = false) {
        auto selected = option;
        if (placement) selected.use_weights_in_host_memory = placement->use_host(models[i], option.use_weights_in_host_memory, prefer_host);
        else if (prefer_host) selected.use_weights_in_host_memory = true;
        return load(models[i], selected, configure, stats, int(i));
    };
    std::vector<std::unique_ptr<ncnn::Net>> resident;
    if (policy == WeightPolicy::Resident)
        for (size_t i=0;i<models.size();++i) resident.push_back(load_block(i));
    stats.peak_loaded_nets = policy == WeightPolicy::Resident ? int(models.size()) : 1;
    // One background Net owns its load counters and allocators. Placement
    // tracing and cache mutation stay on this thread. The future joins on unwind.
    std::future<PreparedBlock> pending;
    Clock::time_point previous_compute_start{}, previous_compute_end{};
    auto finish_prefetch = [&] {
        auto prepared = pending.get();
        stats.load_seconds.insert(stats.load_seconds.end(), prepared.stats.load_seconds.begin(), prepared.stats.load_seconds.end());
        stats.details.insert(stats.details.end(), prepared.stats.details.begin(), prepared.stats.details.end());
        const auto overlap_start = std::max(prepared.start, previous_compute_start);
        const auto overlap_end = std::min(prepared.end, previous_compute_end);
        if (overlap_end > overlap_start)
            stats.prefetch_overlap_seconds += std::chrono::duration<double>(overlap_end - overlap_start).count();
        stats.prefetch_peak_charged_bytes = std::max(stats.prefetch_peak_charged_bytes, prepared.charge);
        return prepared;
    };
    ncnn::VkMat current = input;
    try {
    for (size_t i = 0; i < models.size(); ++i)
    {
        auto acquire_block = [&](bool prefer_host) -> std::unique_ptr<ncnn::Net> {
            if (!pending.valid()) return load_block(i, prefer_host);
            PreparedBlock prepared;
            try { prepared = finish_prefetch(); }
            catch (...) { ++stats.prefetch_skipped; throw; }
            if (prepared.net && prefetch_headroom(*memory, 0))
            {
                ++stats.prefetch_used;
                return std::move(prepared.net);
            }
            ++stats.prefetch_skipped;
            prepared.net.reset();
            return load_block(i, prefer_host);
        };
        WeightSession::Lease lease;
        if (session)
        {
            const auto loaded = stats.load_seconds.size();
            lease = session->acquire(i, models[i], &option, prepared_estimate(models[i]), acquire_block);
            if (stats.load_seconds.size() == loaded) stats.load_seconds.push_back(0);
            stats.peak_loaded_nets = std::max(stats.peak_loaded_nets,
                int(session->stats().cached_nets + (lease.cached() ? 0 : 1)));
        }
        auto streamed = !session && policy == WeightPolicy::Stream ? acquire_block(false) : nullptr;
        auto& net = session ? lease.net() : policy == WeightPolicy::Stream ? *streamed : *resident[i];
        if (memory) set_attention_query_rows(net, memory->query_rows);
        for (const auto* layer : net.layers())
            if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                throw std::runtime_error("Sequence graph contains a compute layer without Vulkan support");
        if (memory && memory->prefetch_bytes && i + 1 < models.size() &&
            (!session || !session->contains(i + 1)))
        {
            const auto estimate = prepared_estimate(models[i + 1]);
            if (estimate <= memory->prefetch_bytes && prefetch_headroom(*memory, estimate))
            {
                auto selected = option;
                selected.use_weights_in_host_memory = placement ?
                    placement->use_host(models[i + 1], option.use_weights_in_host_memory, true, "prefetch") : true;
                if (selected.use_weights_in_host_memory)
                {
                    const auto files = models[i + 1];
                    const int index = int(i + 1);
                    const auto budget = memory->prefetch_bytes;
                    pending = std::async(std::launch::async, [files, selected, device, index, collect_details, budget] {
                        PreparedBlock prepared;
                        prepared.start = Clock::now();
                        prepared.stats.collect_details = collect_details;
                        prepared.net = load(files, selected, [device](ncnn::Net& n) { n.set_vulkan_device(device); }, prepared.stats, index);
                        const auto charge = dit_host_weight_inspector(device)(*prepared.net);
                        prepared.charge = charge.value_or(0);
                        // Unknown graphs, actual GPU fallback and oversized
                        // results are discarded; their synchronous path remains.
                        if (!charge || *charge > budget) prepared.net.reset();
                        prepared.end = Clock::now();
                        return prepared;
                    });
                    ++stats.prefetch_started;
                    stats.peak_loaded_nets = std::max(stats.peak_loaded_nets,
                        int(session ? session->stats().cached_nets + (lease.cached() ? 1 : 2) : 2));
                }
                else ++stats.prefetch_skipped;
            }
            else ++stats.prefetch_skipped;
        }
        const auto start = Clock::now();
        previous_compute_start = start;
        const auto submissions_before = attention_internal_submissions(net);
        {
            auto extractor = net.create_extractor();
            extractor.set_blob_vkallocator(option.blob_vkallocator);
            extractor.set_workspace_vkallocator(option.workspace_vkallocator ? option.workspace_vkallocator : option.blob_vkallocator);
            extractor.set_staging_vkallocator(option.staging_vkallocator);
            check(extractor.input("in0", current), "input device activation");
            for (size_t k = 0; k < constants.size(); ++k)
                check(extractor.input(("in" + std::to_string(k + 1)).c_str(), constants[k]), "input device constant");
            ncnn::VkCompute command(device);
            ncnn::VkMat next;
            try {
                check(extractor.extract("out0", next, command), "extract device block output");
                check(command.submit_and_wait(), "complete block before releasing weights");
            } catch (...) {
                if(stats.collect_details) stats.details.push_back({int(i),"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});
                throw;
            }
            stats.compute_submissions += 1 + attention_internal_submissions(net) - submissions_before;
            current = next;
            stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
            if(stats.collect_details) stats.details.push_back({int(i),"extract_compute_composite","complete",stats.compute_seconds.back()});
            if (observer) observer("block-" + std::to_string(i), current);
        }
        previous_compute_end = Clock::now();
        reclaim_completed(option.blob_vkallocator);
        if (option.workspace_vkallocator != option.blob_vkallocator)
            reclaim_completed(option.workspace_vkallocator);
        if (session) lease.complete();
        if (streamed) { const auto destroy=Clock::now();streamed.reset();if(stats.collect_details) stats.details.push_back({int(i),"net_destroy","complete",std::chrono::duration<double>(Clock::now()-destroy).count()}); }
    }
    }
    catch (...)
    {
        const auto original = std::current_exception();
        previous_compute_end = Clock::now();
        if (pending.valid())
        {
            ++stats.prefetch_skipped;
            // Preserve the current execution error; a joined background load
            // must not replace it, or disappear from completed-load accounting.
            try { auto unused = finish_prefetch(); } catch (...) {}
        }
        std::rethrow_exception(original);
    }
    return current;
}
#endif
ncnn::Mat run_block_sequence(const std::vector<std::string>& models, const ncnn::Mat& input,
    const std::vector<ncnn::Mat>& constants, const ncnn::Option& option,
    WeightPolicy policy, BlockSequenceStats& stats, const CpuStageObserver& observer)
{
    check_request(models.size(), constants.size(), policy);
    return run_block_sequence(component_files(models, "block"), input, constants, option, policy, stats, observer);
}
#if NCNN_VULKAN
ncnn::VkMat run_block_sequence(const std::vector<std::string>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats,
    const VulkanStageObserver& observer, WeightPlacement* placement, WeightSession* session,
    const MemoryExecution* memory)
{
    check_request(models.size(), constants.size(), policy);
    return run_block_sequence(component_files(models, "block"), input, constants, device, option, policy, stats, observer, placement, session, memory);
}
#endif
} // namespace ernie

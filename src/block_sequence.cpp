// SPDX-License-Identifier: MIT
#include "block_sequence.h"
#include "ernie_gelu.h"
#include "ernie_attention.h"
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace ernie {
namespace {
using Clock = std::chrono::steady_clock;
void check(int result, const char* action)
{
    if (result) throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result));
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
        if (streamed) { const auto destroy=Clock::now();streamed.reset();if(stats.collect_details) stats.details.push_back({int(i),"net_destroy","complete",std::chrono::duration<double>(Clock::now()-destroy).count()}); }
    }
    return current;
}

#if NCNN_VULKAN
ncnn::VkMat run_block_sequence(const std::vector<ComponentFiles>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats,
    const VulkanStageObserver& observer)
{
    check_request(models.size(), constants.size(), policy);
    if (!device || !option.use_vulkan_compute || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Vulkan sequence requires device and session allocators");
    const bool collect_details=stats.collect_details;
    stats = {};
    stats.collect_details=collect_details;
    auto configure = [device](ncnn::Net& net) { net.set_vulkan_device(device); };
    std::vector<std::unique_ptr<ncnn::Net>> resident;
    if (policy == WeightPolicy::Resident)
        for (size_t i=0;i<models.size();++i) resident.push_back(load(models[i], option, configure, stats, int(i)));
    stats.peak_loaded_nets = policy == WeightPolicy::Resident ? int(models.size()) : 1;
    ncnn::VkMat current = input;
    for (size_t i = 0; i < models.size(); ++i)
    {
        auto streamed = policy == WeightPolicy::Stream ? load(models[i], option, configure, stats, int(i)) : nullptr;
        auto& net = policy == WeightPolicy::Stream ? *streamed : *resident[i];
        for (const auto* layer : net.layers())
            if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                throw std::runtime_error("Sequence graph contains a compute layer without Vulkan support");
        const auto start = Clock::now();
        auto extractor = net.create_extractor();
        extractor.set_blob_vkallocator(option.blob_vkallocator);
        extractor.set_workspace_vkallocator(option.workspace_vkallocator ? option.workspace_vkallocator : option.blob_vkallocator);
        extractor.set_staging_vkallocator(option.staging_vkallocator);
        check(extractor.input("in0", current), "input device activation");
        for (size_t k = 0; k < constants.size(); ++k)
            check(extractor.input(("in" + std::to_string(k + 1)).c_str(), constants[k]), "input device constant");
        ncnn::VkCompute command(device);
        ncnn::VkMat next;
        try { check(extractor.extract("out0", next, command), "extract device block output");
            check(command.submit_and_wait(), "complete block before releasing weights"); }
        catch (...) { if(stats.collect_details) stats.details.push_back({int(i),"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()}); throw; }
        stats.compute_submissions += 1 + attention_internal_submissions(net);
        current = next;
        stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        if(stats.collect_details) stats.details.push_back({int(i),"extract_compute_composite","complete",stats.compute_seconds.back()});
        if (observer) observer("block-" + std::to_string(i), current);
        if (streamed) { const auto destroy=Clock::now();streamed.reset();if(stats.collect_details) stats.details.push_back({int(i),"net_destroy","complete",std::chrono::duration<double>(Clock::now()-destroy).count()}); }
        // command, extractor, then the streamed Net are destroyed in this order.
        // current remains owned by the caller's blob allocator.
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
    const VulkanStageObserver& observer)
{
    check_request(models.size(), constants.size(), policy);
    return run_block_sequence(component_files(models, "block"), input, constants, device, option, policy, stats, observer);
}
#endif
} // namespace ernie

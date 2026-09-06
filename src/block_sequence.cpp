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
std::unique_ptr<ncnn::Net> load(const std::string& directory, const ncnn::Option& option,
                              Configure&& configure, BlockSequenceStats& stats)
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
    const auto path = std::filesystem::path(directory);
    check(net->load_param((path / "block.ncnn.param").string().c_str()), "load block graph");
    check(net->load_model((path / "block.ncnn.bin").string().c_str()), "load block weights");
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

ncnn::Mat run_block_sequence(const std::vector<std::string>& models, const ncnn::Mat& input,
    const std::vector<ncnn::Mat>& constants, const ncnn::Option& option,
    WeightPolicy policy, BlockSequenceStats& stats, const CpuStageObserver& observer)
{
    check_request(models.size(), constants.size(), policy);
    if (option.use_vulkan_compute) throw std::invalid_argument("CPU sequence requires CPU options");
    stats = {};
    auto configure = [](ncnn::Net&) {};
    std::vector<std::unique_ptr<ncnn::Net>> resident;
    if (policy == WeightPolicy::Resident)
        for (const auto& path : models) resident.push_back(load(path, option, configure, stats));
    stats.peak_loaded_nets = policy == WeightPolicy::Resident ? int(models.size()) : 1;
    ncnn::Mat current = input;
    for (size_t i = 0; i < models.size(); ++i)
    {
        auto streamed = policy == WeightPolicy::Stream ? load(models[i], option, configure, stats) : nullptr;
        auto& net = policy == WeightPolicy::Stream ? *streamed : *resident[i];
        const auto start = Clock::now();
        auto extractor = net.create_extractor();
        check(extractor.input("in0", current), "input activation");
        for (size_t k = 0; k < constants.size(); ++k)
            check(extractor.input(("in" + std::to_string(k + 1)).c_str(), constants[k]), "input constant");
        ncnn::Mat next;
        check(extractor.extract("out0", next), "extract block output");
        // Extractor's default output conversion gives pack1 FP32. Mat retains
        // storage independently of Net, so it survives streamed destruction.
        current = next;
        stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        if (observer) observer("block-" + std::to_string(i), current);
    }
    return current;
}

#if NCNN_VULKAN
ncnn::VkMat run_block_sequence(const std::vector<std::string>& models, const ncnn::VkMat& input,
    const std::vector<ncnn::VkMat>& constants, const ncnn::VulkanDevice* device,
    const ncnn::Option& option, WeightPolicy policy, BlockSequenceStats& stats,
    const VulkanStageObserver& observer)
{
    check_request(models.size(), constants.size(), policy);
    if (!device || !option.use_vulkan_compute || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Vulkan sequence requires device and session allocators");
    stats = {};
    auto configure = [device](ncnn::Net& net) { net.set_vulkan_device(device); };
    std::vector<std::unique_ptr<ncnn::Net>> resident;
    if (policy == WeightPolicy::Resident)
        for (const auto& path : models) resident.push_back(load(path, option, configure, stats));
    stats.peak_loaded_nets = policy == WeightPolicy::Resident ? int(models.size()) : 1;
    ncnn::VkMat current = input;
    for (size_t i = 0; i < models.size(); ++i)
    {
        auto streamed = policy == WeightPolicy::Stream ? load(models[i], option, configure, stats) : nullptr;
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
        check(extractor.extract("out0", next, command), "extract device block output");
        check(command.submit_and_wait(), "complete block before releasing weights");
        stats.compute_submissions += 1 + attention_internal_submissions(net);
        current = next;
        stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        if (observer) observer("block-" + std::to_string(i), current);
        // command, extractor, then the streamed Net are destroyed in this order.
        // current remains owned by the caller's blob allocator.
    }
    return current;
}
#endif
} // namespace ernie

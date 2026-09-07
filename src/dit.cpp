// SPDX-License-Identifier: MIT
#include "dit.h"
#include "ernie_gelu.h"
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
void load(ncnn::Net& net, const ComponentFiles& files)
{
    check(register_layers(net), "register layers");
    load_component(net, files);
}
std::vector<ncnn::Mat> head(const ComponentFiles& files, const std::vector<ncnn::Mat>& inputs,
                            size_t outputs, const ncnn::Option& option, DitStats& stats, const char* component)
{
    auto start=Clock::now();
    auto net=std::make_unique<ncnn::Net>();
    net->opt = option;
    try { check(register_layers(*net), "register layers");load_component_param(*net,files); } catch (...) { if(stats.collect_details) stats.details.push_back({component,"net_setup_param","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
    if(stats.collect_details) stats.details.push_back({component,"net_setup_param","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    start=Clock::now();try { load_component_model(*net,files); } catch (...) { if(stats.collect_details) stats.details.push_back({component,"model_load_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
    if(stats.collect_details) stats.details.push_back({component,"model_load_composite","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    start=Clock::now();
    std::vector<ncnn::Mat> result(outputs);
    try {
        auto extractor = net->create_extractor();
        for (size_t i = 0; i < inputs.size(); ++i)
            check(extractor.input(("in" + std::to_string(i)).c_str(), inputs[i]), "input head tensor");
        for (size_t i = 0; i < outputs; ++i)
            check(extractor.extract(("out" + std::to_string(i)).c_str(), result[i]), "extract head tensor");
    } catch (...) { if(stats.collect_details) stats.details.push_back({component,"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
    if(stats.collect_details) stats.details.push_back({component,"extract_compute_composite","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    start=Clock::now();net.reset();if(stats.collect_details) stats.details.push_back({component,"net_destroy","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    return result;
}
#if NCNN_VULKAN
std::vector<ncnn::VkMat> head(const ComponentFiles& files, const std::vector<ncnn::VkMat>& inputs,
                              size_t outputs, const ncnn::VulkanDevice* device, const ncnn::Option& option,
                              DitStats& stats,const char* component, WeightPlacement* placement)
{
    auto start=Clock::now();
    auto net=std::make_unique<ncnn::Net>();
    net->opt = option;
    net->opt.blob_vkallocator = net->opt.workspace_vkallocator = net->opt.staging_vkallocator = nullptr;
    if (placement) net->opt.use_weights_in_host_memory = placement->use_host(files, option.use_weights_in_host_memory);
    net->set_vulkan_device(device);
    try { check(register_layers(*net), "register layers");load_component_param(*net,files); } catch (...) { if(stats.collect_details) stats.details.push_back({component,"net_setup_param","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
    if(stats.collect_details) stats.details.push_back({component,"net_setup_param","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    start=Clock::now();try { load_component_model(*net,files); } catch (...) { if(stats.collect_details) stats.details.push_back({component,"model_load_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
    if(stats.collect_details) stats.details.push_back({component,"model_load_composite","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    for (const auto* layer : net->layers())
        if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
            throw std::runtime_error("Head has a compute layer without Vulkan support");
    start=Clock::now();std::vector<ncnn::VkMat> result(outputs);
    {
    ncnn::VkCompute command(device);
    auto extractor = net->create_extractor();
    extractor.set_blob_vkallocator(option.blob_vkallocator);
    extractor.set_workspace_vkallocator(option.workspace_vkallocator ? option.workspace_vkallocator : option.blob_vkallocator);
    extractor.set_staging_vkallocator(option.staging_vkallocator);
    try {
        for (size_t i = 0; i < inputs.size(); ++i)
            check(extractor.input(("in" + std::to_string(i)).c_str(), inputs[i]), "input device head tensor");
        for (size_t i = 0; i < outputs; ++i)
            check(extractor.extract(("out" + std::to_string(i)).c_str(), result[i], command), "extract device head tensor");
        check(command.submit_and_wait(), "finish head before releasing weights");
    } catch (...) { if(stats.collect_details) stats.details.push_back({component,"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
    if(stats.collect_details) stats.details.push_back({component,"extract_compute_composite","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    }
    start=Clock::now();net.reset();if(stats.collect_details) stats.details.push_back({component,"net_destroy","complete",std::chrono::duration<double>(Clock::now()-start).count()});
    return result;
}

ncnn::VkMat modulation(const ncnn::VkMat& source, const ncnn::VulkanDevice* device,
                        ncnn::VkCompute& command, const ncnn::Option& option)
{
    ncnn::VkMat result;
    device->convert_packing(source, result, 1, command, option);
    // The exported input head has one 4096-wide row in a singleton channel.
    // Removing only that singleton is a dense view; no channel padding moves.
    if (result.empty() || result.elempack != 1 || result.w != 4096
        || result.h != 1 || result.d != 1 || result.c != 1)
        throw std::runtime_error("Unexpected modulation layout");
    result.dims = 2;
    result.cstep = 4096;
    return result;
}
#endif
} // namespace

ncnn::Mat run_dit(const ComponentFiles& input_head, const std::vector<ComponentFiles>& blocks,
    const ComponentFiles& output_head, const std::vector<ncnn::Mat>& inputs,
    const ncnn::Option& option, DitStats& stats, const CpuStageObserver& observer)
{
    if (inputs.size() != 6 || option.use_vulkan_compute) throw std::invalid_argument("Require six CPU DiT inputs");
    const bool collect_details=stats.collect_details;
    stats = {};
    stats.collect_details=collect_details;
    stats.blocks.collect_details=collect_details;
    auto start = Clock::now();
    auto projected = head(input_head, {inputs[0], inputs[1], inputs[2]}, 8, option,stats,"input-head");
    stats.input_head_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (observer)
        for (size_t i = 0; i < projected.size(); ++i) observer("head-" + std::to_string(i), projected[i]);
    std::vector<ncnn::Mat> constants;
    for (size_t i = 2; i < 8; ++i)
    {
        auto value = projected[i].reshape(4096, 1);
        if (value.empty()) throw std::runtime_error("Unexpected modulation layout");
        constants.push_back(value);
    }
    constants.insert(constants.end(), inputs.begin() + 3, inputs.end());
    const auto current = run_block_sequence(blocks, projected[0], constants, option, WeightPolicy::Stream, stats.blocks, observer);
    start = Clock::now();
    auto result = head(output_head, {current, projected[1]}, 1, option,stats,"output-head")[0];
    stats.output_head_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

#if NCNN_VULKAN
ncnn::VkMat run_dit(const ComponentFiles& input_head, const std::vector<ComponentFiles>& blocks,
    const ComponentFiles& output_head, const std::vector<ncnn::VkMat>& inputs,
    const ncnn::VulkanDevice* device, const ncnn::Option& option, DitStats& stats,
    const VulkanStageObserver& observer, WeightPlacement* placement, WeightSession* session)
{
    if (inputs.size() != 6 || !device || !option.use_vulkan_compute
        || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Require six device DiT inputs and session allocators");
    const bool collect_details=stats.collect_details;
    stats = {};
    stats.collect_details=collect_details;
    stats.blocks.collect_details=collect_details;
    auto start = Clock::now();
    auto projected = head(input_head, {inputs[0], inputs[1], inputs[2]}, 8, device, option,stats,"input-head", placement);
    stats.input_head_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (observer)
        for (size_t i = 0; i < projected.size(); ++i) observer("head-" + std::to_string(i), projected[i]);
    std::vector<ncnn::VkMat> constants;
    {
        ncnn::VkCompute command(device);
        for (size_t i = 2; i < 8; ++i) constants.push_back(modulation(projected[i], device, command, option));
        check(command.submit_and_wait(), "prepare shared modulation");
    }
    constants.insert(constants.end(), inputs.begin() + 3, inputs.end());
    const auto current = run_block_sequence(blocks, projected[0], constants, device, option, WeightPolicy::Stream, stats.blocks, observer, placement, session);
    start = Clock::now();
    auto result = head(output_head, {current, projected[1]}, 1, device, option,stats,"output-head", placement)[0];
    stats.output_head_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}
#endif
ncnn::Mat run_dit(const std::string& input_head, const std::vector<std::string>& blocks,
    const std::string& output_head, const std::vector<ncnn::Mat>& inputs,
    const ncnn::Option& option, DitStats& stats, const CpuStageObserver& observer)
{
    return run_dit(component_files(std::filesystem::u8path(input_head), "head"), component_files(blocks, "block"),
                   component_files(std::filesystem::u8path(output_head), "head"), inputs, option, stats, observer);
}
#if NCNN_VULKAN
ncnn::VkMat run_dit(const std::string& input_head, const std::vector<std::string>& blocks,
    const std::string& output_head, const std::vector<ncnn::VkMat>& inputs,
    const ncnn::VulkanDevice* device, const ncnn::Option& option, DitStats& stats,
    const VulkanStageObserver& observer, WeightPlacement* placement, WeightSession* session)
{
    return run_dit(component_files(std::filesystem::u8path(input_head), "head"), component_files(blocks, "block"),
                   component_files(std::filesystem::u8path(output_head), "head"), inputs, device, option, stats, observer, placement, session);
}
#endif
} // namespace ernie

// SPDX-License-Identifier: MIT
#include "dit.h"
#include "ernie_gelu.h"
#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace ernie {
namespace {
using Clock = std::chrono::steady_clock;
void check(int result, const char* action)
{
    if (result) throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result));
}
void load(ncnn::Net& net, const std::string& path)
{
    check(register_layers(net), "register layers");
    const auto directory = std::filesystem::path(path);
    check(net.load_param((directory / "head.ncnn.param").string().c_str()), "load head graph");
    check(net.load_model((directory / "head.ncnn.bin").string().c_str()), "load head weights");
}
std::vector<ncnn::Mat> head(const std::string& path, const std::vector<ncnn::Mat>& inputs,
                            size_t outputs, const ncnn::Option& option)
{
    ncnn::Net net;
    net.opt = option;
    load(net, path);
    auto extractor = net.create_extractor();
    for (size_t i = 0; i < inputs.size(); ++i)
        check(extractor.input(("in" + std::to_string(i)).c_str(), inputs[i]), "input head tensor");
    std::vector<ncnn::Mat> result(outputs);
    for (size_t i = 0; i < outputs; ++i)
        check(extractor.extract(("out" + std::to_string(i)).c_str(), result[i]), "extract head tensor");
    return result;
}
#if NCNN_VULKAN
std::vector<ncnn::VkMat> head(const std::string& path, const std::vector<ncnn::VkMat>& inputs,
                              size_t outputs, const ncnn::VulkanDevice* device, const ncnn::Option& option)
{
    ncnn::Net net;
    net.opt = option;
    net.opt.blob_vkallocator = net.opt.workspace_vkallocator = net.opt.staging_vkallocator = nullptr;
    net.set_vulkan_device(device);
    load(net, path);
    for (const auto* layer : net.layers())
        if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
            throw std::runtime_error("Head has a compute layer without Vulkan support");
    std::vector<ncnn::VkMat> result(outputs);
    ncnn::VkCompute command(device);
    auto extractor = net.create_extractor();
    extractor.set_blob_vkallocator(option.blob_vkallocator);
    extractor.set_workspace_vkallocator(option.workspace_vkallocator ? option.workspace_vkallocator : option.blob_vkallocator);
    extractor.set_staging_vkallocator(option.staging_vkallocator);
    for (size_t i = 0; i < inputs.size(); ++i)
        check(extractor.input(("in" + std::to_string(i)).c_str(), inputs[i]), "input device head tensor");
    for (size_t i = 0; i < outputs; ++i)
        check(extractor.extract(("out" + std::to_string(i)).c_str(), result[i], command), "extract device head tensor");
    check(command.submit_and_wait(), "finish head before releasing weights");
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

ncnn::Mat run_dit(const std::string& input_head, const std::vector<std::string>& blocks,
    const std::string& output_head, const std::vector<ncnn::Mat>& inputs,
    const ncnn::Option& option, DitStats& stats, const CpuStageObserver& observer)
{
    if (inputs.size() != 6 || option.use_vulkan_compute) throw std::invalid_argument("Require six CPU DiT inputs");
    stats = {};
    auto start = Clock::now();
    auto projected = head(input_head, {inputs[0], inputs[1], inputs[2]}, 8, option);
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
    auto result = head(output_head, {current, projected[1]}, 1, option)[0];
    stats.output_head_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

#if NCNN_VULKAN
ncnn::VkMat run_dit(const std::string& input_head, const std::vector<std::string>& blocks,
    const std::string& output_head, const std::vector<ncnn::VkMat>& inputs,
    const ncnn::VulkanDevice* device, const ncnn::Option& option, DitStats& stats,
    const VulkanStageObserver& observer)
{
    if (inputs.size() != 6 || !device || !option.use_vulkan_compute
        || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Require six device DiT inputs and session allocators");
    stats = {};
    auto start = Clock::now();
    auto projected = head(input_head, {inputs[0], inputs[1], inputs[2]}, 8, device, option);
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
    const auto current = run_block_sequence(blocks, projected[0], constants, device, option, WeightPolicy::Stream, stats.blocks, observer);
    start = Clock::now();
    auto result = head(output_head, {current, projected[1]}, 1, device, option)[0];
    stats.output_head_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}
#endif
}

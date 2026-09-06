// SPDX-License-Identifier: MIT
#include "text_encoder.h"
#include "ernie_text_down.h"
#include "ernie_gelu.h"
#include "ernie_attention.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ernie
{
namespace
{
using Clock = std::chrono::steady_clock;
void check(int result, const char *action)
{
    if (result)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result));
}
void bucket_check(int bucket)
{
    if (bucket < 1 || bucket > 2048)
        throw std::invalid_argument("Text bucket must be in [1,2048]");
}
void load(ncnn::Net &net, const ComponentFiles &files, TextDownMode mode=TextDownMode::Gemm, int bucket=0)
{
    check(register_layers(net), "register text normalization");
    if (mode == TextDownMode::Vector)
    {
        const auto derived = vector_text_down_graph(files.param_text, bucket);
        validate_text_down_weights(files.weight_path);
        check(register_text_down(net), "register vector text down");
        load_component(net, {derived, files.weight_path});
    }
    else load_component(net, files);
}
void request_check(size_t models, size_t constants)
{
    if (models < 1 || models > 25 || constants != 3)
        throw std::invalid_argument("Text path requires 1..25 blocks and cos/sin/causal mask");
}
} // namespace
ncnn::Mat text_embeddings(const std::string &path, const std::vector<uint32_t> &ids, int bucket)
{
    bucket_check(bucket);
    if (ids.empty() || ids.size() > size_t(bucket) ||
        std::filesystem::file_size(path) != 131072ull * 3072 * 2)
        throw std::invalid_argument("Invalid token IDs or BF16 embedding table size");
    ncnn::Mat out(3072, bucket);
    if (out.empty())
        throw std::bad_alloc();
    out.fill(0.f);
    std::ifstream file(path, std::ios::binary);
    unsigned char bytes[3072 * 2];
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (ids[i] >= 131072)
            throw std::invalid_argument("Token ID outside vocabulary");
        file.seekg(size_t(ids[i]) * sizeof(bytes));
        if (!file.read(reinterpret_cast<char *>(bytes), sizeof(bytes)))
            throw std::runtime_error("Embedding row read failed");
        float *target = out.row(int(i));
        for (int j = 0; j < 3072; ++j)
        {
            const uint32_t bits = (uint32_t(bytes[2 * j]) | uint32_t(bytes[2 * j + 1]) << 8) << 16;
            std::memcpy(&target[j], &bits, 4);
        }
    }
    return out;
}
std::vector<ncnn::Mat> text_constants(const std::string &path, int bucket)
{
    bucket_check(bucket);
    if (std::filesystem::file_size(path) != 64 * 4)
        throw std::invalid_argument("Invalid YaRN frequency table size");
    float frequencies[64];
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(frequencies), sizeof(frequencies)))
        throw std::runtime_error("Cannot read YaRN frequencies");
    ncnn::Mat cos(128, bucket), sin(128, bucket), mask(bucket, bucket);
    if (cos.empty() || sin.empty() || mask.empty())
        throw std::bad_alloc();
    for (int t = 0; t < bucket; ++t)
    {
        float *c = cos.row(t);
        float *s = sin.row(t);
        float *m = mask.row(t);
        for (int j = 0; j < 64; ++j)
        {
            if (!std::isfinite(frequencies[j]) || frequencies[j] <= 0.f)
                throw std::runtime_error("Invalid YaRN frequency");
            const float phase = float(t) * frequencies[j];
            c[j] = c[j + 64] = std::cos(phase);
            s[j] = s[j + 64] = std::sin(phase);
        }
        for (int k = 0; k < bucket; ++k)
            m[k] = k > t ? -std::numeric_limits<float>::max() : 0.f;
    }
    return {cos, sin, mask};
}
ncnn::Mat run_text_blocks(const std::vector<ComponentFiles> &models, const ncnn::Mat &input,
                          const std::vector<ncnn::Mat> &constants, const ncnn::Option &option,
                          BlockSequenceStats &stats, TextDownMode down_mode)
{
    request_check(models.size(), constants.size());
    if (option.use_vulkan_compute)
        throw std::invalid_argument("CPU text path requires CPU options");
    if (down_mode!=TextDownMode::Gemm && down_mode!=TextDownMode::Vector)
        throw std::invalid_argument("Unknown text down mode");
    if (down_mode==TextDownMode::Vector)
    {
        if (input.dims!=2 || input.w!=3072 || input.elemsize!=4u || input.elempack!=1 ||
            (input.h!=64 && input.h!=2048) || option.use_fp16_storage || option.use_fp16_packed ||
            option.use_fp16_arithmetic || option.use_bf16_storage || option.use_bf16_packed)
            throw std::invalid_argument("Vector text down requires reviewed CPU FP32 layout");
        for (size_t i=0;i<constants.size();++i)
            if (constants[i].dims!=2 || constants[i].w!=(i==2?input.h:128) ||
                constants[i].h!=input.h || constants[i].elemsize!=4u || constants[i].elempack!=1)
                throw std::invalid_argument("Vector text constants have wrong layout");
    }
    const bool collect_details=stats.collect_details;
    stats = {};
    stats.collect_details=collect_details;
    stats.peak_loaded_nets = 1;
    ncnn::Mat current = input;
    for (size_t block=0;block<models.size();++block)
    {
        const auto &path=models[block];
        auto start = Clock::now();
        auto net=std::make_unique<ncnn::Net>();
        net->opt = option;
        try {
            check(register_layers(*net), "register text normalization");
            if (down_mode==TextDownMode::Vector) { const auto derived=vector_text_down_graph(path.param_text,input.h);validate_text_down_weights(path.weight_path);check(register_text_down(*net),"register vector text down");load_component_param(*net,{derived,path.weight_path}); }
            else load_component_param(*net,path);
        } catch (...) { if(stats.collect_details)stats.details.push_back({int(block),"net_setup_param","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
        if(stats.collect_details) stats.details.push_back({int(block),"net_setup_param","complete",std::chrono::duration<double>(Clock::now()-start).count()});
        auto model_start=Clock::now();try { load_component_model(*net,path); }
        catch (...) { if(stats.collect_details)stats.details.push_back({int(block),"model_load_composite","failed",std::chrono::duration<double>(Clock::now()-model_start).count()});throw; }
        if(stats.collect_details) stats.details.push_back({int(block),"model_load_composite","complete",std::chrono::duration<double>(Clock::now()-model_start).count()});
        stats.load_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        start = Clock::now();
        {
        try {
        auto extractor = net->create_extractor();
        check(extractor.input("in0", current), "input text activation");
        for (size_t i = 0; i < constants.size(); ++i)
            check(extractor.input(("in" + std::to_string(i + 1)).c_str(), constants[i]),
                  "input text constant");
        ncnn::Mat next;
        check(extractor.extract("out0", next), "extract text hidden state");
        current = next;
        stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        if(stats.collect_details) stats.details.push_back({int(block),"extract_compute_composite","complete",stats.compute_seconds.back()});
        } catch (...) { if(stats.collect_details)stats.details.push_back({int(block),"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
        }
        const auto destroy=Clock::now();net.reset();
        if(stats.collect_details) stats.details.push_back({int(block),"net_destroy","complete",std::chrono::duration<double>(Clock::now()-destroy).count()});
    }
    return current;
}
#if NCNN_VULKAN
ncnn::VkMat run_text_blocks(const std::vector<ComponentFiles> &models, const ncnn::VkMat &input,
                            const std::vector<ncnn::VkMat> &constants, const ncnn::VulkanDevice *device,
                            const ncnn::Option &option, BlockSequenceStats &stats)
{
    request_check(models.size(), constants.size());
    if (!device || !option.use_vulkan_compute || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Device text path requires session allocators");
    const bool collect_details=stats.collect_details;
    stats = {};
    stats.collect_details=collect_details;
    stats.peak_loaded_nets = 1;
    ncnn::VkMat current = input;
    for (size_t block=0;block<models.size();++block)
    {
        const auto &path=models[block];
        auto start = Clock::now();
        auto net=std::make_unique<ncnn::Net>();
        net->opt = option;
        net->opt.blob_vkallocator = net->opt.workspace_vkallocator = net->opt.staging_vkallocator = nullptr;
        net->set_vulkan_device(device);
        try { check(register_layers(*net), "register text normalization");load_component_param(*net,path); }
        catch (...) { if(stats.collect_details)stats.details.push_back({int(block),"net_setup_param","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
        if(stats.collect_details) stats.details.push_back({int(block),"net_setup_param","complete",std::chrono::duration<double>(Clock::now()-start).count()});
        auto model_start=Clock::now();try { load_component_model(*net,path); }
        catch (...) { if(stats.collect_details)stats.details.push_back({int(block),"model_load_composite","failed",std::chrono::duration<double>(Clock::now()-model_start).count()});throw; }
        if(stats.collect_details) stats.details.push_back({int(block),"model_load_composite","complete",std::chrono::duration<double>(Clock::now()-model_start).count()});
        for (const auto *layer : net->layers())
            if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                throw std::runtime_error("Text graph contains a compute layer without Vulkan support");
        stats.load_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        start = Clock::now();
        {
        try {
        auto extractor = net->create_extractor();
        extractor.set_blob_vkallocator(option.blob_vkallocator);
        extractor.set_workspace_vkallocator(option.workspace_vkallocator ? option.workspace_vkallocator
                                                                         : option.blob_vkallocator);
        extractor.set_staging_vkallocator(option.staging_vkallocator);
        check(extractor.input("in0", current), "input device text activation");
        for (size_t i = 0; i < constants.size(); ++i)
            check(extractor.input(("in" + std::to_string(i + 1)).c_str(), constants[i]),
                  "input device text constant");
        ncnn::VkCompute command(device);
        ncnn::VkMat next;
        check(extractor.extract("out0", next, command), "extract device text hidden state");
        check(command.submit_and_wait(), "finish text block before releasing weights");
        current = next;
        stats.compute_submissions += 1 + attention_internal_submissions(*net);
        stats.compute_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
        if(stats.collect_details) stats.details.push_back({int(block),"extract_compute_composite","complete",stats.compute_seconds.back()});
        } catch (...) { if(stats.collect_details)stats.details.push_back({int(block),"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-start).count()});throw; }
        }
        const auto destroy=Clock::now();net.reset();
        if(stats.collect_details) stats.details.push_back({int(block),"net_destroy","complete",std::chrono::duration<double>(Clock::now()-destroy).count()});
    }
    return current;
}
#endif
ncnn::Mat run_text_blocks(const std::vector<std::string> &models, const ncnn::Mat &input,
                          const std::vector<ncnn::Mat> &constants, const ncnn::Option &option,
                          BlockSequenceStats &stats, TextDownMode down_mode)
{
    request_check(models.size(), constants.size());
    return run_text_blocks(component_files(models, "text"), input, constants, option, stats, down_mode);
}
#if NCNN_VULKAN
ncnn::VkMat run_text_blocks(const std::vector<std::string> &models, const ncnn::VkMat &input,
                            const std::vector<ncnn::VkMat> &constants, const ncnn::VulkanDevice *device,
                            const ncnn::Option &option, BlockSequenceStats &stats)
{
    request_check(models.size(), constants.size());
    return run_text_blocks(component_files(models, "text"), input, constants, device, option, stats);
}
#endif
} // namespace ernie

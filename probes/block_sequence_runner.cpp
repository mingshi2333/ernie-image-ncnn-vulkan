// SPDX-License-Identifier: MIT
#include "dit.h"
#if NCNN_VULKAN
#include "pipelinecache.h"
#endif
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

static ncnn::Mat read(const fs::path& path, int width, int height)
{
    const size_t bytes = size_t(width) * height * sizeof(float);
    if (fs::file_size(path) != bytes) throw std::runtime_error("Input byte count differs: " + path.string());
    ncnn::Mat result(width, height);
    if (result.empty()) throw std::bad_alloc();
    std::ifstream file(path, std::ios::binary);
    if (!file.read(static_cast<char*>(result.data), bytes)) throw std::runtime_error("Cannot read " + path.string());
    return result;
}

static void write_stage(const fs::path& path, const ncnn::Mat& value)
{
    if (value.empty() || value.elempack != 1 || value.elemsize != 4u || fs::exists(path))
        throw std::runtime_error("Invalid stage tensor or existing output");
    std::ofstream file(path, std::ios::binary);
    for (int c = 0; c < value.c; ++c)
        file.write(reinterpret_cast<const char*>(static_cast<const float*>(value.channel(c))),
                   size_t(value.w) * value.h * value.d * sizeof(float));
    if (!file) throw std::runtime_error("Cannot write stage tensor");
}

int main(int argc, char** argv)
{
    std::vector<std::string> models;
    fs::path fixture, output, input_head, output_head, trace;
    std::string backend = "cpu", precision = "fp32", policy = "stream";
    int tokens = 0, status = 0;
    int width = 0, height = 0, text_tokens = 0;
    bool host_weights = false;
    bool shared_pipeline_cache = true;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--host-weights") { host_weights = true; continue; }
            if (flag == "--isolated-pipeline-cache") { shared_pipeline_cache = false; continue; }
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "--model") models.push_back(value);
            else if (flag == "--fixture") fixture = value;
            else if (flag == "--output") output = value;
            else if (flag == "--input-head") input_head = value;
            else if (flag == "--output-head") output_head = value;
            else if (flag == "--trace-dir") trace = value;
            else if (flag == "--backend") backend = value;
            else if (flag == "--precision") precision = value;
            else if (flag == "--policy") policy = value;
            else if (flag == "--tokens" || flag == "--width" || flag == "--height" || flag == "--text-tokens")
            {
                size_t consumed = 0;
                const int number = std::stoi(value, &consumed);
                if (consumed != value.size()) throw std::invalid_argument("Invalid token count");
                if (flag == "--tokens") tokens = number;
                else if (flag == "--width") width = number;
                else if (flag == "--height") height = number;
                else text_tokens = number;
            }
            else throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (models.empty() || fixture.empty() || output.empty() || tokens < 1 || tokens > 6144
            || (backend != "cpu" && backend != "vulkan") || (policy != "stream" && policy != "resident")
            || (precision != "fp32" && precision != "fp16" && precision != "bf16")
            || (backend == "cpu" && precision != "fp32"))
            throw std::invalid_argument("Require repeated --model DIR, --fixture DIR, --tokens N, --output FILE "
                "[--backend cpu|vulkan] [--precision fp32|fp16|bf16] [--policy stream|resident] [--host-weights]");
        if (fs::exists(output)) throw std::invalid_argument("Output exists; use a new path");
        if (!trace.empty())
        {
            if (fs::exists(trace)) throw std::invalid_argument("Trace exists; use a new path");
            fs::create_directories(trace);
        }
        const bool dit = !input_head.empty() || !output_head.empty();
        if (dit && (input_head.empty() || output_head.empty() || policy != "stream"
            || width < 1 || width > 256 || height < 1 || height > 256 || text_tokens < 1
            || text_tokens > 2048 || width * height + text_tokens != tokens))
            throw std::invalid_argument("DiT mode requires both heads, matching grid/text dimensions and streaming");
        const auto input = dit ? read(fixture / "in0.f32", width * height, 128).reshape(width, height, 128)
                               : read(fixture / "in0.f32", 4096, tokens);
        std::vector<ncnn::Mat> constants;
        if (dit)
        {
            constants.push_back(read(fixture / "in1.f32", 3072, text_tokens));
            constants.push_back(read(fixture / "in2.f32", 4096, 1).reshape(4096));
        }
        else
            for (int i = 1; i <= 6; ++i) constants.push_back(read(fixture / ("in" + std::to_string(i) + ".f32"), 4096, 1));
        constants.push_back(read(fixture / (dit ? "in3.f32" : "in7.f32"), 128, tokens));
        constants.push_back(read(fixture / (dit ? "in4.f32" : "in8.f32"), 128, tokens));
        constants.push_back(read(fixture / (dit ? "in5.f32" : "in9.f32"), tokens, tokens));
        ncnn::Option option;
        option.num_threads = 4;
        option.use_vulkan_compute = backend == "vulkan";
        option.use_fp16_storage = precision == "fp16";
        option.use_bf16_storage = precision == "bf16";
        option.use_fp16_packed = option.use_fp16_arithmetic = option.use_bf16_packed = false;
        option.use_weights_in_host_memory = host_weights;
        const auto weight_policy = policy == "stream" ? ernie::WeightPolicy::Stream : ernie::WeightPolicy::Resident;
        ernie::BlockSequenceStats stats;
        ernie::DitStats dit_stats;
        ncnn::Mat result;
        if (backend == "cpu")
        {
            ernie::CpuStageObserver observer;
            if (!trace.empty()) observer = [&](const std::string& name, const ncnn::Mat& value) {
                write_stage(trace / (name + ".f32"), value);
            };
            if (dit)
            {
                std::vector<ncnn::Mat> inputs{input};
                inputs.insert(inputs.end(), constants.begin(), constants.end());
                result = ernie::run_dit(input_head.string(), models, output_head.string(), inputs, option, dit_stats, observer);
                stats = dit_stats.blocks;
            }
            else result = ernie::run_block_sequence(models, input, constants, option, weight_policy, stats, observer);
        }
        else
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1) throw std::runtime_error("No Vulkan device");
            const int index = ncnn::get_default_gpu_index();
            const auto& info = ncnn::get_gpu_info(index);
            if ((precision == "fp16" && !info.support_fp16_storage()) ||
                (precision == "bf16" && !info.support_bf16_storage()))
                throw std::runtime_error("Requested device storage precision is unavailable");
            const auto* device = ncnn::get_gpu_device(index);
            // Net otherwise creates and destroys its own cache for every block.
            // Retain compiled pipelines across streamed Nets and both heads.
            ncnn::PipelineCache pipeline_cache(device);
            option.pipeline_cache = shared_pipeline_cache ? &pipeline_cache : nullptr;
            ncnn::VkBlobAllocator blob_allocator(device);
            ncnn::VkStagingAllocator staging_allocator(device);
            option.blob_vkallocator = option.workspace_vkallocator = &blob_allocator;
            option.staging_vkallocator = &staging_allocator;
            ncnn::VkMat gpu_input;
            std::vector<ncnn::VkMat> gpu_constants(constants.size());
            {
                ncnn::VkCompute upload(device);
                upload.record_upload(input, gpu_input, option);
                for (size_t i = 0; i < constants.size(); ++i) upload.record_upload(constants[i], gpu_constants[i], option);
                if (upload.submit_and_wait()) throw std::runtime_error("Initial upload failed");
            }
            ncnn::VkMat gpu_result;
            ernie::VulkanStageObserver observer;
            if (!trace.empty()) observer = [&](const std::string& name, const ncnn::VkMat& value) {
                ncnn::Mat host;
                ncnn::Option plain = option;
                plain.use_packing_layout = false;
                ncnn::VkCompute download(device);
                download.record_download(value, host, plain);
                if (download.submit_and_wait()) throw std::runtime_error("Stage download failed");
                write_stage(trace / (name + ".f32"), host);
            };
            if (dit)
            {
                std::vector<ncnn::VkMat> inputs{gpu_input};
                inputs.insert(inputs.end(), gpu_constants.begin(), gpu_constants.end());
                gpu_result = ernie::run_dit(input_head.string(), models, output_head.string(), inputs, device, option, dit_stats, observer);
                stats = dit_stats.blocks;
            }
            else gpu_result = ernie::run_block_sequence(models, gpu_input, gpu_constants, device, option, weight_policy, stats, observer);
            ncnn::VkCompute download(device);
            ncnn::Option plain = option;
            plain.use_packing_layout = false;
            download.record_download(gpu_result, result, plain);
            if (download.submit_and_wait()) throw std::runtime_error("Final download failed");
#else
            throw std::runtime_error("Built without Vulkan");
#endif
        }
        if (result.elempack != 1 || result.elemsize != 4u
            || (dit ? result.dims != 3 || result.w != width || result.h != height || result.c != 128
                    : result.dims != 2 || result.w != 4096 || result.h != tokens))
            throw std::runtime_error("Unexpected sequence output layout");
        if (output.has_parent_path()) fs::create_directories(output.parent_path());
        std::ofstream file(output, std::ios::binary);
        for (int channel = 0; channel < result.c; ++channel)
            file.write(reinterpret_cast<const char*>(static_cast<const float*>(result.channel(channel))),
                       size_t(result.w) * result.h * sizeof(float));
        if (!file) throw std::runtime_error("Cannot write output");
        std::cout << std::setprecision(9) << "{\"backend\":\"" << backend << "\",\"precision\":\"" << precision
                  << "\",\"policy\":\"" << policy << "\",\"blocks\":" << models.size()
                  << ",\"peak_loaded_nets\":" << stats.peak_loaded_nets
                  << ",\"host_weights\":" << (host_weights ? "true" : "false")
                  << ",\"shared_pipeline_cache\":" << (backend == "vulkan" && shared_pipeline_cache ? "true" : "false")
                  << ",\"compute_submissions\":" << stats.compute_submissions << ",\"load_seconds\":[";
        for (size_t i = 0; i < stats.load_seconds.size(); ++i) std::cout << (i ? "," : "") << stats.load_seconds[i];
        std::cout << "],\"compute_seconds\":[";
        for (size_t i = 0; i < stats.compute_seconds.size(); ++i) std::cout << (i ? "," : "") << stats.compute_seconds[i];
        std::cout << "],\"dit_heads\":" << (dit ? "true" : "false")
                  << ",\"input_head_seconds\":" << dit_stats.input_head_seconds
                  << ",\"output_head_seconds\":" << dit_stats.output_head_seconds << "}\n";
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; status = 1; }
#if NCNN_VULKAN
    if (backend == "vulkan") ncnn::destroy_gpu_instance();
#endif
    return status;
}

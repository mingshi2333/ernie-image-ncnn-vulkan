// SPDX-License-Identifier: MIT
#include "ernie_gelu.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
static void check(int value, const char* action)
{
    if (value) throw std::runtime_error(std::string(action) + " failed: " + std::to_string(value));
}
static ncnn::Mat read(const fs::path& path, ncnn::Mat value)
{
    if (value.empty()) throw std::bad_alloc();
    const size_t plane = size_t(value.w) * value.h * value.d;
    if (fs::file_size(path) != plane * value.c * sizeof(float))
        throw std::runtime_error("Input byte count differs: " + path.string());
    std::ifstream file(path, std::ios::binary);
    for (int c = 0; c < value.c; ++c)
        if (!file.read(reinterpret_cast<char*>(static_cast<float*>(value.channel(c))), plane * sizeof(float)))
            throw std::runtime_error("Cannot read input");
    return value;
}
static void write(const fs::path& path, const ncnn::Mat& value, size_t count)
{
    if (value.empty() || value.elempack != 1 || value.elemsize != 4u
        || size_t(value.w) * value.h * value.d * value.c != count)
        throw std::runtime_error("Unexpected output layout");
    if (fs::exists(path)) throw std::runtime_error("Output already exists");
    std::ofstream file(path, std::ios::binary);
    const size_t plane = size_t(value.w) * value.h * value.d;
    for (int c = 0; c < value.c; ++c)
        file.write(reinterpret_cast<const char*>(static_cast<const float*>(value.channel(c))), plane * sizeof(float));
    if (!file) throw std::runtime_error("Cannot write output");
}

int main(int argc, char** argv)
{
    fs::path model, fixture, output;
    std::string component, backend = "cpu", precision = "fp32";
    std::string vae_convolution = "sgemm";
    int width = 0, height = 0, text_tokens = 0, status = 0;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (++i == argc) throw std::invalid_argument("Missing argument value");
            const std::string value = argv[i];
            if (flag == "--model") model = value;
            else if (flag == "--fixture") fixture = value;
            else if (flag == "--output") output = value;
            else if (flag == "--component") component = value;
            else if (flag == "--backend") backend = value;
            else if (flag == "--precision") precision = value;
            else if (flag == "--vae-convolution") vae_convolution = value;
            else if (flag == "--width" || flag == "--height" || flag == "--text-tokens")
            {
                size_t consumed = 0;
                const int number = std::stoi(value, &consumed);
                if (consumed != value.size()) throw std::invalid_argument("Invalid dimension");
                if (flag == "--width") width = number;
                else if (flag == "--height") height = number;
                else text_tokens = number;
            }
            else throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (model.empty() || fixture.empty() || output.empty() || fs::exists(output)
            || width < 1 || width > 256 || height < 1 || height > 256 || text_tokens < (component == "vae" ? 0 : 1)
            || text_tokens > 2048 || (component != "vae" && width * height + text_tokens > 6144)
            || (component != "input" && component != "output" && component != "vae")
            || (backend != "cpu" && backend != "vulkan")
            || (precision != "fp32" && precision != "fp16" && precision != "bf16")
            || (backend == "cpu" && precision != "fp32")
            || (vae_convolution != "sgemm" && vae_convolution != "direct"))
            throw std::invalid_argument("Require --model DIR --fixture DIR --output NEWDIR --component input|output "
                                       "--width N --height N --text-tokens N [--backend cpu|vulkan] [--precision fp32|fp16|bf16]");
        const int tokens = width * height + text_tokens;
        std::vector<ncnn::Mat> inputs;
        std::vector<size_t> counts;
        if (component == "vae")
        {
            inputs.push_back(read(fixture / "in0.f32", ncnn::Mat(width, height, 32)));
            counts.push_back(size_t(width) * height * 64 * 3);
        }
        else if (component == "input")
        {
            inputs.push_back(read(fixture / "in0.f32", ncnn::Mat(width, height, 128)));
            inputs.push_back(read(fixture / "in1.f32", ncnn::Mat(3072, text_tokens)));
            inputs.push_back(read(fixture / "in2.f32", ncnn::Mat(4096)));
            counts.assign(8, 4096);
            counts[0] = size_t(tokens) * 4096;
        }
        else
        {
            inputs.push_back(read(fixture / "in0.f32", ncnn::Mat(4096, tokens)));
            inputs.push_back(read(fixture / "in1.f32", ncnn::Mat(4096)));
            counts.push_back(size_t(width) * height * 128);
        }
        ncnn::Option option;
        option.num_threads = 4;
        option.use_vulkan_compute = backend == "vulkan";
        option.use_fp16_storage = precision == "fp16";
        option.use_bf16_storage = precision == "bf16";
        option.use_fp16_packed = option.use_fp16_arithmetic = option.use_bf16_packed = false;
        if (component == "vae" && backend == "cpu") option.use_winograd_convolution = false;
        if (component == "vae" && backend == "cpu") option.use_sgemm_convolution = vae_convolution == "sgemm";
        std::vector<ncnn::Mat> outputs(counts.size());
        const auto start = std::chrono::steady_clock::now();
        {
            ncnn::Net net;
            net.opt = option;
#if NCNN_VULKAN
            if (option.use_vulkan_compute)
            {
                ncnn::create_gpu_instance();
                if (ncnn::get_gpu_count() < 1) throw std::runtime_error("No Vulkan device");
                net.set_vulkan_device(ncnn::get_default_gpu_index());
                const auto& info = net.vulkan_device()->info;
                if ((precision == "fp16" && !info.support_fp16_storage())
                    || (precision == "bf16" && !info.support_bf16_storage()))
                    throw std::runtime_error("Storage precision is unavailable");
            }
#else
            if (option.use_vulkan_compute) throw std::runtime_error("Built without Vulkan");
#endif
            check(ernie::register_layers(net), "register layers");
            check(net.load_param((model / "head.ncnn.param").c_str()), "load graph");
            check(net.load_model((model / "head.ncnn.bin").c_str()), "load weights");
            if (!option.use_vulkan_compute)
            {
                auto extractor = net.create_extractor();
                for (size_t i = 0; i < inputs.size(); ++i)
                    check(extractor.input(("in" + std::to_string(i)).c_str(), inputs[i]), "input");
                for (size_t i = 0; i < outputs.size(); ++i)
                    check(extractor.extract(("out" + std::to_string(i)).c_str(), outputs[i]), "extract");
            }
#if NCNN_VULKAN
            else
            {
                for (const auto* layer : net.layers())
                    if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                        throw std::runtime_error("Graph requires a layer without Vulkan support: " + layer->type);
                const auto* device = net.vulkan_device();
                ncnn::VkBlobAllocator blob(device);
                ncnn::VkStagingAllocator staging(device);
                option.blob_vkallocator = option.workspace_vkallocator = &blob;
                option.staging_vkallocator = &staging;
                std::vector<ncnn::VkMat> gpu_inputs(inputs.size()), gpu_outputs(outputs.size());
                ncnn::VkCompute command(device);
                auto extractor = net.create_extractor();
                extractor.set_blob_vkallocator(&blob);
                extractor.set_workspace_vkallocator(&blob);
                extractor.set_staging_vkallocator(&staging);
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    command.record_upload(inputs[i], gpu_inputs[i], option);
                    check(extractor.input(("in" + std::to_string(i)).c_str(), gpu_inputs[i]), "device input");
                }
                ncnn::Option plain = option;
                plain.use_packing_layout = false;
                for (size_t i = 0; i < outputs.size(); ++i)
                {
                    check(extractor.extract(("out" + std::to_string(i)).c_str(), gpu_outputs[i], command), "device extract");
                    command.record_download(gpu_outputs[i], outputs[i], plain);
                }
                check(command.submit_and_wait(), "complete head");
            }
#endif
        }
        fs::create_directories(output);
        for (size_t i = 0; i < outputs.size(); ++i)
            write(output / ("out" + std::to_string(i) + ".f32"), outputs[i], counts[i]);
        std::cout << "{\"outputs\":" << outputs.size() << ",\"elapsed_seconds\":"
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "}\n";
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; status = 1; }
#if NCNN_VULKAN
    if (backend == "vulkan") ncnn::destroy_gpu_instance();
#endif
    return status;
}

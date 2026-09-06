// SPDX-License-Identifier: MIT
#include "denoiser.h"
#if NCNN_VULKAN
#include "pipelinecache.h"
#endif
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace fs = std::filesystem;
static ncnn::Mat read(const fs::path &path, int w, int h, int c = 1)
{
    const size_t plane_bytes = size_t(w) * h * sizeof(float);
    if (fs::file_size(path) != plane_bytes * c)
        throw std::runtime_error("Wrong input size: " + path.string());
    ncnn::Mat value = c == 1 ? ncnn::Mat(w, h) : ncnn::Mat(w, h, c);
    if (value.empty())
        throw std::bad_alloc();
    std::ifstream file(path, std::ios::binary);
    for (int k = 0; k < c; ++k)
        if (!file.read(static_cast<char *>(value.channel(k).data), plane_bytes))
            throw std::runtime_error("Input read failed");
    return value;
}
static void write(const fs::path &path, const ncnn::Mat &value)
{
    if (fs::exists(path))
        throw std::runtime_error("Refusing to overwrite " + path.string());
    if (value.empty() || value.elempack != 1 || value.elemsize != 4u)
        throw std::runtime_error("Expected FP32 output");
    std::ofstream file(path, std::ios::binary);
    for (int c = 0; c < value.c; ++c)
        file.write(static_cast<const char *>(value.channel(c).data), size_t(value.w) * value.h * value.d * 4);
    if (!file)
        throw std::runtime_error("Output write failed");
}
static double sum(const std::vector<double> &xs) { return std::accumulate(xs.begin(), xs.end(), 0.); }

int main(int argc, char **argv)
{
    std::vector<std::string> block_directories;
    std::string input_directory, output_directory;
    fs::path fixture, output;
    std::string backend = "cpu", precision = "fp32";
    int w = 0, h = 0, text = 0, steps = 8, status = 0;
    float only_timestep = -1;
    bool trace = false, features_only = false;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--trace")
            {
                trace = true;
                continue;
            }
            if (i + 1 == argc)
                throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "--model")
                block_directories.push_back(value);
            else if (flag == "--input-head")
                input_directory = value;
            else if (flag == "--output-head")
                output_directory = value;
            else if (flag == "--fixture")
                fixture = value;
            else if (flag == "--output")
                output = value;
            else if (flag == "--backend")
                backend = value;
            else if (flag == "--precision")
                precision = value;
            else if (flag == "--timestep")
            {
                size_t used = 0;
                only_timestep = std::stof(value, &used);
                if (used != value.size())
                    throw std::invalid_argument("Invalid timestep");
                features_only = true;
            }
            else if (flag == "--width" || flag == "--height" || flag == "--text-tokens" || flag == "--steps")
            {
                size_t used = 0;
                const int number = std::stoi(value, &used);
                if (used != value.size())
                    throw std::invalid_argument("Invalid integer");
                if (flag == "--width")
                    w = number;
                else if (flag == "--height")
                    h = number;
                else if (flag == "--text-tokens")
                    text = number;
                else
                    steps = number;
            }
            else
                throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (output.empty() || fs::exists(output))
            throw std::invalid_argument("Use a new --output directory");
        if (features_only)
        {
            const auto features = ernie::timestep_features(only_timestep);
            fs::create_directories(output);
            write(output / "features.f32", features);
            return 0;
        }
        if (fixture.empty() || w < 1 || w > 256 || h < 1 || h > 256 || text < 1 || text > 2048 ||
            w * h + text > 6144 || block_directories.empty() || block_directories.size() > 36 ||
            input_directory.empty() || output_directory.empty() || steps < 1 || steps > 1000 ||
            (backend != "cpu" && backend != "vulkan") ||
            (precision != "fp32" && precision != "fp16" && precision != "bf16") ||
            (backend == "cpu" && precision != "fp32"))
            throw std::invalid_argument(
                "Require heads, blocks, fixture, dimensions and cpu/fp32 or vulkan precision");
        ernie::DenoiseModel model{
            ernie::component_files(fs::path(input_directory), "head"),
            ernie::component_files(fs::path(output_directory), "head"),
            ernie::component_files(block_directories, "block")};
        const int tokens = w * h + text;
        const auto initial = read(fixture / "in0.f32", w, h, 128);
        const std::vector<ncnn::Mat> constants{
            read(fixture / "in1.f32", 3072, text), read(fixture / "in3.f32", 128, tokens),
            read(fixture / "in4.f32", 128, tokens), read(fixture / "in5.f32", tokens, tokens)};
        fs::create_directories(output);
        ncnn::Option option;
        option.num_threads = 4;
        option.use_vulkan_compute = backend == "vulkan";
        option.use_fp16_storage = precision == "fp16";
        option.use_bf16_storage = precision == "bf16";
        option.use_fp16_packed = option.use_fp16_arithmetic = option.use_bf16_packed = false;
        std::vector<ernie::DenoiseStepStats> stats;
        auto progress = [&](size_t index)
        {
            const auto &step = stats.at(index);
            std::cout << std::setprecision(12) << "{\"step\":" << index << ",\"timestep\":" << step.timestep
                      << ",\"delta\":" << step.delta << ",\"elapsed_seconds\":" << step.elapsed_seconds
                      << ",\"block_load_seconds\":" << sum(step.dit.blocks.load_seconds)
                      << ",\"block_compute_seconds\":" << sum(step.dit.blocks.compute_seconds)
                      << ",\"input_head_seconds\":" << step.dit.input_head_seconds
                      << ",\"output_head_seconds\":" << step.dit.output_head_seconds << "}" << std::endl;
            if (trace)
                write(output / ("features-" + std::to_string(index) + ".f32"),
                      ernie::timestep_features(step.timestep));
        };
        ncnn::Mat result;
        if (backend == "cpu")
            result = ernie::denoise(model, initial, constants, steps, option, stats,
                                    [&](size_t i, const ncnn::Mat &prediction, const ncnn::Mat &sample)
                                    {
                                        if (trace)
                                        {
                                            write(output / ("prediction-" + std::to_string(i) + ".f32"),
                                                  prediction);
                                            write(output / ("step-" + std::to_string(i) + ".f32"), sample);
                                        }
                                        progress(i);
                                    });
        else
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1)
                throw std::runtime_error("No Vulkan device");
            const auto index = ncnn::get_default_gpu_index();
            const auto &info = ncnn::get_gpu_info(index);
            if ((precision == "fp16" && !info.support_fp16_storage()) ||
                (precision == "bf16" && !info.support_bf16_storage()))
                throw std::runtime_error("Unsupported storage precision");
            const auto *device = ncnn::get_gpu_device(index);
            ncnn::PipelineCache cache(device);
            ncnn::VkBlobAllocator blobs(device);
            ncnn::VkStagingAllocator staging(device);
            option.pipeline_cache = &cache;
            option.blob_vkallocator = option.workspace_vkallocator = &blobs;
            option.staging_vkallocator = &staging;
            ncnn::Option high = option;
            high.use_fp16_storage = high.use_bf16_storage = false;
            high.use_packing_layout = false;
            ncnn::VkMat gpu_initial;
            std::vector<ncnn::VkMat> gpu_constants(constants.size());
            {
                ncnn::VkCompute command(device);
                ncnn::VkMat uploaded;
                command.record_upload(initial, uploaded, high);
                device->convert_packing(uploaded, gpu_initial, 1, 1, command, high);
                for (size_t i = 0; i < constants.size(); ++i)
                    command.record_upload(constants[i], gpu_constants[i], option);
                if (command.submit_and_wait())
                    throw std::runtime_error("Initial upload failed");
            }
            const auto gpu_result =
                ernie::denoise(model, gpu_initial, gpu_constants, steps, device, option, stats,
                               [&](size_t i, const ncnn::VkMat &prediction, const ncnn::VkMat &sample)
                               {
                                   if (trace)
                                   {
                                       ncnn::Mat p, x;
                                       ncnn::VkCompute download(device);
                                       download.record_download(prediction, p, high);
                                       download.record_download(sample, x, high);
                                       if (download.submit_and_wait())
                                           throw std::runtime_error("Trace download failed");
                                       write(output / ("prediction-" + std::to_string(i) + ".f32"), p);
                                       write(output / ("step-" + std::to_string(i) + ".f32"), x);
                                   }
                                   progress(i);
                               });
            ncnn::VkCompute download(device);
            download.record_download(gpu_result, result, high);
            if (download.submit_and_wait())
                throw std::runtime_error("Final download failed");
#else
            throw std::runtime_error("Built without Vulkan");
#endif
        }
        write(output / "final.f32", result);
        std::cout << "{\"complete\":true,\"steps\":" << stats.size() << ",\"blocks\":" << block_directories.size()
                  << ",\"backend\":\"" << backend << "\",\"dit_storage\":\"" << precision
                  << "\",\"scheduler_storage\":\"fp32\",\"trace_downloads\":" << (trace ? "true" : "false")
                  << "}\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
#if NCNN_VULKAN
    if (backend == "vulkan")
        ncnn::destroy_gpu_instance();
#endif
    return status;
}

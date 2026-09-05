// SPDX-License-Identifier: MIT
#include "latent_ops.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static ncnn::Mat read(const fs::path& path, int width, int height = 0, int channels = 0)
{
    ncnn::Mat result = channels ? ncnn::Mat(width, height, channels) : ncnn::Mat(width);
    if (result.empty()) throw std::bad_alloc();
    const int plane = channels ? width * height : width;
    const int count = channels ? channels : 1;
    if (fs::file_size(path) != size_t(plane) * count * sizeof(float))
        throw std::runtime_error("Input size mismatch: " + path.string());
    std::ifstream file(path, std::ios::binary);
    for (int c = 0; c < count; ++c)
    {
        float* target = channels ? result.channel(c) : static_cast<float*>(result);
        if (!file.read(reinterpret_cast<char*>(target), size_t(plane) * sizeof(float)))
            throw std::runtime_error("Cannot read: " + path.string());
    }
    return result;
}

static void write(const fs::path& path, const ncnn::Mat& result)
{
    if (result.dims != 3 || result.elempack != 1 || result.elemsize != 4u)
        throw std::runtime_error("Unexpected output format");
    std::ofstream file(path, std::ios::binary);
    for (int c = 0; c < result.c; ++c)
        file.write(reinterpret_cast<const char*>(static_cast<const float*>(result.channel(c))),
                   size_t(result.w) * result.h * sizeof(float));
    if (!file) throw std::runtime_error("Cannot write: " + path.string());
}

static int number(const std::string& value)
{
    size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("Invalid integer: " + value);
    return result;
}

int main(int argc, char** argv)
{
    std::string backend = "cpu";
    bool batched = false;
    fs::path fixture, output;
    int width = 0, height = 0, steps = 8, status = 0;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--batched") { batched = true; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "--fixture") fixture = value;
            else if (flag == "--output") output = value;
            else if (flag == "--width") width = number(value);
            else if (flag == "--height") height = number(value);
            else if (flag == "--steps") steps = number(value);
            else if (flag == "--backend") backend = value;
            else throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (fixture.empty() || output.empty() || width < 1 || height < 1 || width > 256 || height > 256
            || (backend != "cpu" && backend != "vulkan") || (batched && backend != "vulkan"))
            throw std::invalid_argument("Require --fixture DIR --output NEW_DIR --width N --height N "
                                        "[--steps 8] [--backend cpu|vulkan] [--batched]");
        if (fs::exists(output)) throw std::invalid_argument("Use a new output directory");
        const auto schedule = ernie::FlowSchedule::turbo(steps);
        // The probe retains traces, so bound its memory separately from the API.
        if (uint64_t(width) * height * 128 * steps > 64u * 1024u * 1024u)
            throw std::invalid_argument("Probe trace exceeds 256 MiB per tensor series");
        fs::create_directories(output);
        auto sample = read(fixture / "initial.f32", width, height, 128);
        const auto mean = read(fixture / "mean.f32", 128);
        const auto variance = read(fixture / "variance.f32", 128);
        std::vector<ncnn::Mat> predictions, results(steps);
        for (int step = 0; step < steps; ++step)
            predictions.push_back(read(fixture / ("prediction-" + std::to_string(step) + ".f32"), width, height, 128));
        ncnn::Mat unpacked;
        int submissions = 0;
        if (backend == "cpu")
        {
            for (int step = 0; step < steps; ++step)
            {
                ernie::euler_step(sample, predictions[step], schedule.delta(step), results[step]);
                sample = results[step];
            }
            unpacked = ernie::unpack_for_vae(sample, mean, variance);
        }
        else
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1) throw std::runtime_error("No Vulkan device available");
            const auto* device = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
            ncnn::VkBlobAllocator blob_allocator(device);
            ncnn::VkStagingAllocator staging_allocator(device);
            ernie::VulkanLatentOps ops(device);
            ncnn::Option option;
            option.use_packing_layout = false;
            option.use_fp16_storage = option.use_fp16_packed = option.use_fp16_arithmetic = false;
            option.use_bf16_storage = option.use_bf16_packed = false;
            option.blob_vkallocator = option.workspace_vkallocator = &blob_allocator;
            option.staging_vkallocator = &staging_allocator;
            ncnn::VkMat gpu_sample, gpu_mean, gpu_variance, gpu_unpacked;
            std::vector<ncnn::VkMat> gpu_predictions(steps), gpu_results(steps);
            std::vector<ncnn::VkMat> upload_buffers(steps + 3);
            {
                ncnn::VkCompute upload(device);
                // record_upload chooses pack4 independently of use_packing_layout.
                // Normalize explicitly on device for this pack1-only API.
                auto upload_pack1 = [&](int i, const ncnn::Mat& source, ncnn::VkMat& target) {
                    upload.record_upload(source, upload_buffers[i], option);
                    device->convert_packing(upload_buffers[i], target, 1, upload, option);
                };
                upload_pack1(0, sample, gpu_sample);
                upload_pack1(1, mean, gpu_mean);
                upload_pack1(2, variance, gpu_variance);
                for (int step = 0; step < steps; ++step)
                    upload_pack1(step + 3, predictions[step], gpu_predictions[step]);
                if (upload.submit_and_wait()) throw std::runtime_error("Upload failed");
                ++submissions;
            }
            if (batched)
            {
                ncnn::VkCompute command(device);
                for (int step = 0; step < steps; ++step)
                {
                    ops.record_euler(gpu_sample, gpu_predictions[step], schedule.delta(step),
                                     gpu_results[step], command, option);
                    gpu_sample = gpu_results[step];
                }
                ops.record_unpack(gpu_sample, gpu_mean, gpu_variance, gpu_unpacked, command, option);
                for (int step = 0; step < steps; ++step)
                    command.record_download(gpu_results[step], results[step], option);
                command.record_download(gpu_unpacked, unpacked, option);
                if (command.submit_and_wait()) throw std::runtime_error("Batched latent operations failed");
                ++submissions;
            }
            else
            {
                for (int step = 0; step < steps; ++step)
                {
                    ncnn::VkCompute command(device);
                    ops.record_euler(gpu_sample, gpu_predictions[step], schedule.delta(step),
                                     gpu_results[step], command, option);
                    command.record_download(gpu_results[step], results[step], option);
                    if (command.submit_and_wait()) throw std::runtime_error("Euler step failed");
                    gpu_sample = gpu_results[step];
                    ++submissions;
                }
                ncnn::VkCompute command(device);
                ops.record_unpack(gpu_sample, gpu_mean, gpu_variance, gpu_unpacked, command, option);
                command.record_download(gpu_unpacked, unpacked, option);
                if (command.submit_and_wait()) throw std::runtime_error("Unpack failed");
                ++submissions;
            }
#else
            throw std::runtime_error("Built without Vulkan");
#endif
        }
        for (int step = 0; step < steps; ++step)
            write(output / ("step-" + std::to_string(step) + ".f32"), results[step]);
        write(output / "unpacked.f32", unpacked);
        std::cout << std::setprecision(9) << "{\"backend\":\"" << backend << "\",\"precision\":\"fp32\",\"steps\":"
                  << steps << ",\"submissions\":" << submissions << ",\"batched\":" << (batched ? "true" : "false")
                  << ",\"sigmas\":[";
        for (size_t i = 0; i < schedule.sigmas.size(); ++i) std::cout << (i ? "," : "") << schedule.sigmas[i];
        std::cout << "],\"timesteps\":[";
        for (size_t i = 0; i < schedule.timesteps.size(); ++i) std::cout << (i ? "," : "") << schedule.timesteps[i];
        std::cout << "]}\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
#if NCNN_VULKAN
    if (backend == "vulkan") ncnn::destroy_gpu_instance();
#endif
    return status;
}

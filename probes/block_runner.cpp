// SPDX-License-Identifier: MIT
#include "net.h"
#include "ernie_gelu.h"
#include "attention_diagnostic.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void check(int code, const std::string& action)
{
    if (code != 0)
        throw std::runtime_error(action + " failed: " + std::to_string(code));
}

static ncnn::Mat read_tensor(const fs::path& path, int w, int h = 0)
{
    ncnn::Mat data = h ? ncnn::Mat(w, h) : ncnn::Mat(w);
    const size_t bytes = size_t(w) * (h ? h : 1) * sizeof(float);
    if (fs::file_size(path) != bytes)
        throw std::runtime_error("Tensor byte count mismatch: " + path.string());
    std::ifstream file(path, std::ios::binary);
    if (!file.read(static_cast<char*>(data.data), bytes))
        throw std::runtime_error("Cannot read " + path.string());
    return data;
}

int main(int argc, char** argv)
{
    fs::path model, fixture, output;
    std::string backend = "cpu", precision = "fp32";
    int tokens = 0, threads = 4, repeat = 1;
    bool host_weights = false, device_io = false, trace_attention = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string flag = argv[i];
        if (flag == "--host-weights" || flag == "--device-io" || flag == "--trace-attention")
        {
            if (flag == "--host-weights") host_weights = true;
            else if (flag == "--device-io") device_io = true;
            else trace_attention = true;
            continue;
        }
        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for " << flag << '\n';
            return 2;
        }
        const std::string value = argv[++i];
        if (flag == "--model") model = value;
        else if (flag == "--fixture") fixture = value;
        else if (flag == "--output") output = value;
        else if (flag == "--backend") backend = value;
        else if (flag == "--precision") precision = value;
        else if (flag == "--tokens") tokens = std::atoi(value.c_str());
        else if (flag == "--threads") threads = std::atoi(value.c_str());
        else if (flag == "--repeat") repeat = std::atoi(value.c_str());
        else
        {
            std::cerr << "Unknown option " << flag << '\n';
            return 2;
        }
    }
    if (model.empty() || fixture.empty() || output.empty() || tokens < 1 || tokens > 6144 || threads < 1 || repeat < 1 || repeat > 20
        || ((device_io || trace_attention) && backend != "vulkan")
        || (backend != "cpu" && backend != "vulkan")
        || (precision != "fp32" && precision != "fp16" && precision != "bf16"))
    {
        std::cerr << "Usage: ernie-block-runner --model DIR --fixture DIR --tokens N --output FILE "
                     "[--backend cpu|vulkan] [--precision fp32|fp16|bf16] [--threads N] [--host-weights] [--device-io] [--repeat N] [--trace-attention]\n";
        return 2;
    }
    int status = 0;
    try
    {
        std::vector<ncnn::Mat> inputs;
        inputs.push_back(read_tensor(fixture / "in0.f32", 4096, tokens));
        for (int i = 1; i <= 6; ++i)
            inputs.push_back(read_tensor(fixture / ("in" + std::to_string(i) + ".f32"), 4096, 1));
        inputs.push_back(read_tensor(fixture / "in7.f32", 128, tokens));
        inputs.push_back(read_tensor(fixture / "in8.f32", 128, tokens));
        inputs.push_back(read_tensor(fixture / "in9.f32", tokens, tokens));

        AttentionDiagnostic diagnostic;
        ncnn::Net net;
        check(ernie::register_layers(net), "register ERNIE layers");
        net.opt.num_threads = threads;
        net.opt.use_vulkan_compute = backend == "vulkan";
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = precision == "fp16";
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_packed = false;
        net.opt.use_bf16_storage = precision == "bf16";
        net.opt.use_weights_in_host_memory = host_weights;
        if (backend == "vulkan")
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1)
                throw std::runtime_error("No Vulkan device available");
            const int index = ncnn::get_default_gpu_index();
            const auto& info = ncnn::get_gpu_info(index);
            if (precision == "bf16" && !info.support_bf16_storage())
                throw std::runtime_error("Requested BF16 storage is unavailable");
            if (precision == "fp16" && !info.support_fp16_storage())
                throw std::runtime_error("Requested FP16 storage is unavailable");
            net.set_vulkan_device(index);
            if (trace_attention) check(register_attention_diagnostic(net, diagnostic), "register attention diagnostic");
            std::cerr << "Device: " << info.device_name() << '\n';
#else
            throw std::runtime_error("Built without Vulkan");
#endif
        }
        const auto load_start = Clock::now();
        check(net.load_param((model / "block.ncnn.param").string().c_str()), "load graph");
        check(net.load_model((model / "block.ncnn.bin").string().c_str()), "load weights");
        const auto load_end = Clock::now();
        std::vector<std::string> non_vulkan_layers;
        for (const auto* layer : net.layers())
            if (backend == "vulkan" && !layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                non_vulkan_layers.push_back(layer->type + ":" + layer->name);
        if (device_io && !non_vulkan_layers.empty())
            throw std::runtime_error("Device I/O requires all compute layers to support Vulkan");
        double upload_seconds = 0;
        std::vector<double> run_seconds;
#if NCNN_VULKAN
        std::unique_ptr<ncnn::VkBlobAllocator> blob_allocator;
        std::unique_ptr<ncnn::VkStagingAllocator> staging_allocator;
        std::vector<ncnn::VkMat> device_inputs;
        ncnn::Option transfer = net.opt;
        if (device_io)
        {
            blob_allocator = std::make_unique<ncnn::VkBlobAllocator>(net.vulkan_device());
            staging_allocator = std::make_unique<ncnn::VkStagingAllocator>(net.vulkan_device());
            transfer.blob_vkallocator = transfer.workspace_vkallocator = blob_allocator.get();
            transfer.staging_vkallocator = staging_allocator.get();
            device_inputs.resize(inputs.size());
            const auto start = Clock::now();
            ncnn::VkCompute cmd(net.vulkan_device());
            for (size_t i = 0; i < inputs.size(); ++i)
                cmd.record_upload(inputs[i], device_inputs[i], transfer);
            check(cmd.submit_and_wait(), "upload fixture");
            upload_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        }
#endif
        ncnn::Mat result;
        ncnn::Mat first_result;
        double repeat_max_abs = 0;
        for (int iteration = 0; iteration < repeat; ++iteration)
        {
            result.release();
            const auto start = Clock::now();
            ncnn::Extractor ex = net.create_extractor();
#if NCNN_VULKAN
            if (device_io)
            {
                ex.set_blob_vkallocator(blob_allocator.get());
                ex.set_workspace_vkallocator(blob_allocator.get());
                ex.set_staging_vkallocator(staging_allocator.get());
                for (size_t i = 0; i < device_inputs.size(); ++i)
                    check(ex.input(("in" + std::to_string(i)).c_str(), device_inputs[i]), "input VkMat");
                ncnn::VkCompute cmd(net.vulkan_device());
                ncnn::VkMat out_gpu;
                check(ex.extract("out0", out_gpu, cmd), "extract VkMat");
                ncnn::Option download = transfer;
                download.use_packing_layout = false;
                cmd.record_download(out_gpu, result, download);
                check(cmd.submit_and_wait(), "run and download block");
            }
            else
#endif
            {
                for (size_t i = 0; i < inputs.size(); ++i)
                    check(ex.input(("in" + std::to_string(i)).c_str(), inputs[i]), "input Mat");
                check(ex.extract("out0", result), "run block");
            }
            run_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
            if (result.w != 4096 || result.h != tokens || result.elemsize != 4 || result.elempack != 1)
                throw std::runtime_error("Unexpected output layout");
            if (iteration == 0) first_result = result.clone();
            else
                for (int y = 0; y < tokens; ++y)
                    for (int x = 0; x < 4096; ++x)
                        repeat_max_abs = std::max(repeat_max_abs, std::abs(double(result.row(y)[x]) - first_result.row(y)[x]));
        }
        if (repeat_max_abs != 0) throw std::runtime_error("Repeated same-input execution changed output");
        if (result.w != 4096 || result.h != tokens || result.c != 1 || result.elempack != 1 || result.elemsize != 4)
            throw std::runtime_error("Unexpected block output shape: " + std::to_string(result.w) + ","
                                     + std::to_string(result.h) + "," + std::to_string(result.c));
        if (output.has_parent_path()) fs::create_directories(output.parent_path());
        std::ofstream file(output, std::ios::binary);
        for (int y = 0; y < tokens; ++y)
            file.write(reinterpret_cast<const char*>(result.row(y)), 4096 * sizeof(float));
        if (!file) throw std::runtime_error("Failed to write block output");
        std::cout << "{\"backend\":\"" << backend << "\",\"precision\":\"" << precision
                  << "\",\"tokens\":" << tokens << ",\"load_seconds\":"
                  << std::chrono::duration<double>(load_end - load_start).count()
                  << ",\"first_run_seconds\":" << run_seconds[0] << ",\"run_seconds\":[";
        for (size_t i = 0; i < run_seconds.size(); ++i) std::cout << (i ? "," : "") << run_seconds[i];
        std::cout << "],\"host_weights\":" << (host_weights ? "true" : "false")
                  << ",\"device_io\":" << (device_io ? "true" : "false")
                  << ",\"upload_seconds\":" << upload_seconds << ",\"repeat_max_abs\":" << repeat_max_abs
                  << ",\"non_vulkan_compute_layers\":[";
        for (size_t i = 0; i < non_vulkan_layers.size(); ++i)
            std::cout << (i ? "," : "") << '"' << non_vulkan_layers[i] << '"';
        std::cout << "],\"attention_diagnostic\":";
        if (trace_attention)
            std::cout << "{\"calls\":" << diagnostic.calls << ",\"flash_calls\":" << diagnostic.flash_calls
                      << ",\"cooperative_calls\":" << diagnostic.cooperative_calls << ",\"tokens\":" << diagnostic.tokens
                      << ",\"head_dim\":" << diagnostic.head_dim << "}";
        else std::cout << "null";
        std::cout << "}\n";
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

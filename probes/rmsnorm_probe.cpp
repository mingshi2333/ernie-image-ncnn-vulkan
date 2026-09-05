// SPDX-License-Identifier: MIT
#include "ernie_rmsnorm.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const std::string backend = argc > 1 ? argv[1] : "cpu";
    const std::string precision = argc > 2 ? argv[2] : "fp32";
    bool native = false, centered = false;
    for (int i = 3; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--native") native = true;
        else if (std::string(argv[i]) == "--layernorm") centered = true;
        else return 2;
    }
    if ((backend != "cpu" && backend != "vulkan") ||
        (precision != "fp32" && precision != "fp16" && precision != "bf16") ||
        (backend == "cpu" && precision != "fp32")) return 2;
    bool passed = true;
    try
    {
#if NCNN_VULKAN
        if (backend == "vulkan")
        {
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1) { ncnn::destroy_gpu_instance(); return 77; }
            const auto& info = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
            if ((precision == "fp16" && !info.support_fp16_storage()) ||
                (precision == "bf16" && !info.support_bf16_storage()))
            { ncnn::destroy_gpu_instance(); return 77; }
        }
#else
        if (backend == "vulkan") return 77;
#endif
        for (int width : (centered ? std::vector<int>{4096} : std::vector<int>{128, 3072, 4096}))
            for (float scale : {1.f, centered ? 1024.f : 512.f})
            {
                ncnn::Net net;
                if (!native && (centered ? ernie::register_layernorm(net) : ernie::register_rmsnorm(net)))
                    throw std::runtime_error("Registration failed");
                net.opt.use_vulkan_compute = backend == "vulkan";
                net.opt.num_threads = 4;
                net.opt.use_fp16_storage = precision == "fp16";
                net.opt.use_bf16_storage = precision == "bf16";
                net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = net.opt.use_bf16_packed = false;
                const std::string graph = std::string("7767517\n2 2\nInput input 0 1 in0\n")
                    + (centered ? "LayerNorm" : "RMSNorm") + " norm 1 1 in0 out0 0="
                    + std::to_string(width) + " 1=0.000001 2=" + (centered ? "0\n" : "1\n");
                std::vector<float> weight(width);
                for (int i = 0; i < width; ++i) weight[i] = .5f + (i % 8) * .0625f;
                // The memory overload returns bytes consumed on success.
                if (net.load_param_mem(graph.c_str()) || net.load_model(reinterpret_cast<const unsigned char*>(weight.data())) < 0)
                    throw std::runtime_error("Model load failed");
                // Outer hidden norm is 2D, Q/K norm is 3D. All input values and
                // weights are exactly representable in FP32, FP16 and BF16.
                ncnn::Mat input = width == 128 ? ncnn::Mat(width, 5, 8) : ncnn::Mat(width, 8);
                for (int c = 0; c < input.c; ++c)
                {
                    float* values = input.channel(c);
                    for (int row = 0; row < input.h; ++row)
                        for (int x = 0; x < width; ++x)
                            values[row * width + x] = scale * (.5f + ((x + row + c) % 8) * .125f);
                }
                ncnn::Mat output;
                auto ex = net.create_extractor();
                if (ex.input("in0", input) || ex.extract("out0", output)) throw std::runtime_error("Inference failed");
                double maximum = 0;
                for (int c = 0; c < input.c; ++c)
                {
                    const float* values = input.channel(c);
                    const float* actual = output.channel(c);
                    for (int row = 0; row < input.h; ++row)
                    {
                        double squared = 0, mean = 0;
                        if (centered)
                        {
                            for (int x = 0; x < width; ++x) mean += values[row * width + x];
                            mean /= width;
                        }
                        for (int x = 0; x < width; ++x)
                        {
                            const double delta = values[row * width + x] - mean;
                            squared += delta * delta;
                        }
                        const double inverse = 1. / std::sqrt(squared / width + 1e-6);
                        for (int x = 0; x < width; ++x)
                        {
                            const double expected = (values[row * width + x] - mean) * inverse * (centered ? 1. : weight[x]);
                            if (!std::isfinite(actual[row * width + x])) throw std::runtime_error("Non-finite normalization");
                            maximum = std::max(maximum, std::abs(actual[row * width + x] - expected));
                        }
                    }
                }
                const double limit = precision == "fp32" ? 2e-6 : precision == "fp16" ? .002 : .01;
                const bool ok = maximum <= limit;
                passed &= ok;
                std::cout << "{\"backend\":\"" << backend << "\",\"precision\":\"" << precision << "\",\"native\":"
                          << (native ? "true" : "false") << ",\"centered\":" << (centered ? "true" : "false")
                          << ",\"width\":" << width << ",\"scale\":" << scale
                          << ",\"max_abs_error\":" << maximum << ",\"limit\":" << limit
                          << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
            }
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; passed = false; }
#if NCNN_VULKAN
    if (backend == "vulkan") ncnn::destroy_gpu_instance();
#endif
    return passed ? 0 : 1;
}

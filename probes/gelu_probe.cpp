// SPDX-License-Identifier: MIT
#include "ernie_gelu.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    const std::string backend = argc > 1 ? argv[1] : "cpu";
    const std::string precision = argc > 2 ? argv[2] : "fp32";
    if ((backend != "cpu" && backend != "vulkan") ||
        (precision != "fp32" && precision != "fp16" && precision != "bf16")) return 2;
    bool passed = true;
    try
    {
#if NCNN_VULKAN
        if (backend == "vulkan")
        {
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1)
            {
                ncnn::destroy_gpu_instance();
                return 77;
            }
            const auto& info = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
            if ((precision == "fp16" && !info.support_fp16_storage()) ||
                (precision == "bf16" && !info.support_bf16_storage()))
            {
                ncnn::destroy_gpu_instance();
                return 77;
            }
        }
#else
        if (backend == "vulkan") return 77;
#endif
        for (int shape = 0; shape < 2; ++shape)
        {
            ncnn::Net net;
            if (ernie::register_layers(net)) throw std::runtime_error("register failed");
            net.opt.num_threads = 4;
            net.opt.use_vulkan_compute = backend == "vulkan";
            net.opt.use_fp16_storage = precision == "fp16";
            net.opt.use_fp16_packed = false;
            net.opt.use_fp16_arithmetic = false;
            net.opt.use_bf16_storage = precision == "bf16";
            net.opt.use_bf16_packed = false;
            const char* graph = "7767517\n2 2\nInput input 0 1 in0\nErnieGELU activation 1 1 in0 out0\n";
            alignas(4) const unsigned char empty_weights[4] = {};
            if (net.load_param_mem(graph) || net.load_model(empty_weights)) throw std::runtime_error("load failed");
            ncnn::Mat input = shape ? ncnn::Mat(128, 257, 3) : ncnn::Mat(4096, 16);
            const int count = input.w * input.h * input.c;
            for (int c = 0; c < input.c; ++c)
            {
                float* values = input.channel(c);
                for (int i = 0; i < input.w * input.h; ++i)
                    values[i] = -12.f + 24.f * (i + c * input.w * input.h) / (count - 1);
            }
            auto ex = net.create_extractor();
            ncnn::Mat output;
            if (ex.input("in0", input) || ex.extract("out0", output)) throw std::runtime_error("inference failed");
            double max_error = 0;
            for (int c = 0; c < input.c; ++c)
            {
                const float* values = input.channel(c);
                const float* result = output.channel(c);
                for (int i = 0; i < input.w * input.h; ++i)
                {
                    const double x = values[i];
                    const double expected = 0.5 * x * std::erfc(-x / std::sqrt(2.0));
                    if (!std::isfinite(result[i])) throw std::runtime_error("non-finite result");
                    max_error = std::max(max_error, std::abs(result[i] - expected));
                }
            }
            const double limit = precision == "fp32" ? 2e-6 : (precision == "fp16" ? .008 : .064);
            bool ok = max_error <= limit;
            passed = passed && ok;
            std::cout << "{\"backend\":\"" << backend << "\",\"precision\":\"" << precision << "\",\"shape\":" << shape
                      << ",\"samples\":" << count << ",\"max_abs_error\":" << max_error << ",\"limit\":" << limit
                      << ",\"passed\":" << (ok ? "true" : "false") << "}\n";
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        passed = false;
    }
#if NCNN_VULKAN
    if (backend == "vulkan") ncnn::destroy_gpu_instance();
#endif
    return passed ? 0 : 1;
}

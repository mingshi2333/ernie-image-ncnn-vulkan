// SPDX-License-Identifier: MIT
#include "ernie_gelu.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// A constant value column must remain constant under attention, even when
// thousands of individually tiny probabilities contribute to it. Returning to
// a single long FP32 accumulator would fail this regression. Other columns use
// an independent FP64 softmax/dot oracle and include cancellation and masking.
int main()
{
#if !NCNN_VULKAN
    return 77;
#else
    int status = 1;
    try
    {
        ncnn::create_gpu_instance();
        if (ncnn::get_gpu_count() < 1) { ncnn::destroy_gpu_instance(); return 77; }
        constexpr int width = 128, queries = 5, keys = 4160, heads = 2, visible = 4128;
        ncnn::Mat q(width, queries, heads), k(width, keys, heads), v(width, keys, heads), mask(keys, queries);
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < queries; ++i)
                for (int d = 0; d < width; ++d)
                    q.channel(h).row(i)[d] = i == 0 ? 0.f : float(((d * 17 + i * 13 + h) % 41) - 20) / 40.f;
            for (int j = 0; j < keys; ++j)
                for (int d = 0; d < width; ++d)
                {
                    k.channel(h).row(j)[d] = float(((j * 7 + d * 19 + h * 11) % 101) - 50) / 64.f;
                    v.channel(h).row(j)[d] = d == 0 ? 1.f : float(((j * 47 + d * 13 + h * 31) % 257) - 128) / 128.f;
                }
        }
        for (int i = 0; i < queries; ++i)
            for (int j = 0; j < keys; ++j) mask.row(i)[j] = j < visible ? 0.f : -1e30f;
        ncnn::Mat expected(width, queries, heads);
        for (int h = 0; h < heads; ++h)
            for (int i = 0; i < queries; ++i)
            {
                std::vector<double> probabilities(visible);
                double maximum = -1e100, total = 0.;
                for (int j = 0; j < visible; ++j)
                {
                    double score = 0.;
                    for (int d = 0; d < width; ++d) score += double(q.channel(h).row(i)[d]) * k.channel(h).row(j)[d];
                    probabilities[j] = score / std::sqrt(double(width));
                    maximum = std::max(maximum, probabilities[j]);
                }
                for (double& p : probabilities) { p = std::exp(p - maximum); total += p; }
                for (int d = 0; d < width; ++d)
                {
                    double sum = 0.;
                    for (int j = 0; j < visible; ++j) sum += probabilities[j] * v.channel(h).row(j)[d];
                    expected.channel(h).row(i)[d] = float(sum / total);
                }
            }
        ncnn::Mat actual;
        {
            ncnn::Net net;
            net.opt.num_threads = 4;
            net.opt.use_vulkan_compute = true;
            net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
            net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
            if (ernie::register_layers(net)) throw std::runtime_error("Register layers failed");
            const char* graph = "7767517\n5 5\nInput q 0 1 q\nInput k 0 1 k\nInput v 0 1 v\nInput mask 0 1 mask\nSDPA attention 4 1 q k v mask out 5=1\n";
            alignas(4) const unsigned char weights[4] = {};
            if (net.load_param_mem(graph) || net.load_model(weights)) throw std::runtime_error("Load graph failed");
            auto ex = net.create_extractor();
            if (ex.input("q", q) || ex.input("k", k) || ex.input("v", v) || ex.input("mask", mask)
                || ex.extract("out", actual)) throw std::runtime_error("Attention execution failed");
        }
        if (actual.w != width || actual.h != queries || actual.c != heads || actual.elempack != 1 || actual.elemsize != 4u)
            throw std::runtime_error("Unexpected attention output shape");
        double error_sum = 0., reference_sum = 0., maximum = 0., constant_error = 0.;
        for (int h = 0; h < heads; ++h)
            for (int i = 0; i < queries; ++i)
            {
                constant_error = std::max(constant_error, std::abs(double(actual.channel(h).row(i)[0]) - 1.));
                for (int d = 0; d < width; ++d)
                {
                    const double a = actual.channel(h).row(i)[d], b = expected.channel(h).row(i)[d];
                    if (!std::isfinite(a)) throw std::runtime_error("Nonfinite attention output");
                    error_sum += (a-b)*(a-b); reference_sum += b*b;
                    maximum = std::max(maximum, std::abs(a-b));
                }
            }
        const double nrmse = std::sqrt(error_sum / reference_sum);
        const bool passed = maximum <= 3e-6 && constant_error <= 5e-7 && nrmse <= 5e-7;
        std::cout << std::setprecision(12) << "{\"scope\":\"FP32 long attention, FP64 oracle and constant preservation\",\"keys\":" << keys
                  << ",\"max_abs_error\":" << maximum << ",\"constant_error\":" << constant_error << ",\"nrmse\":" << nrmse
                  << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
        status = passed ? 0 : 1;
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
    ncnn::destroy_gpu_instance();
    return status;
#endif
}

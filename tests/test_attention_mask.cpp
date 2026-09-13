// SPDX-License-Identifier: MIT
#include "conditioning.h"
#include "ernie_attention.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

void run_case(bool vulkan, const std::string& precision, int queries, int keys, int mask_heads, int width = 128)
{
    constexpr int heads = 2;
    ncnn::Mat q(width, queries, heads), k(width, keys, 1), v(width, keys, 1);
    for (int h = 0; h < heads; ++h)
        for (int i = 0; i < queries; ++i)
            for (int d = 0; d < width; ++d)
                q.channel(h).row(i)[d] = float((i * 3 + d * 7 + h) % 17 - 8) / 32.f;
    for (int j = 0; j < keys; ++j)
        for (int d = 0; d < width; ++d)
        {
            k.channel(0).row(j)[d] = float((j * 7 + d) % 19 - 9) / 32.f;
            v.channel(0).row(j)[d] = float((j * 3 + d * 5) % 23 - 11) / 16.f;
        }
    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = precision == "fp16";
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = net.opt.use_bf16_packed = precision == "bf16";
    require(ernie::register_attention(net, true) == 0, "Register SDPA");
    const char* graph = "7767517\n5 5\nInput q 0 1 q\nInput k 0 1 k\nInput v 0 1 v\n"
                        "Input mask 0 1 mask\nSDPA attention 4 1 q k v mask out 5=1\n";
    alignas(4) const unsigned char weights[4] = {};
    require(!net.load_param_mem(graph) && !net.load_model(weights), "Load SDPA");
    ernie::set_attention_query_rows(net, 64);
    // Reuse one pipeline with two different masks. Test odd tiles, per-head
    // broadcast, GQA, bounded FP32 and low-storage flash against dense input.
    for (int shift : {0, 3})
    {
        ncnn::Mat dense, row;
        if (mask_heads == 1) { dense.create(keys, queries); row.create(keys, 1); }
        else { dense.create(keys, queries, mask_heads); row.create(keys, 1, mask_heads); }
        for (int h = 0; h < mask_heads; ++h)
            for (int j = 0; j < keys; ++j)
            {
                const float value = j < keys - 4 - shift - h ? 0.f : -1e30f;
                row.channel(h).row(0)[j] = value;
                for (int i = 0; i < queries; ++i) dense.channel(h).row(i)[j] = value;
            }
        ncnn::Mat baseline;
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            auto ex = net.create_extractor();
            require(!ex.input("q", q) && !ex.input("k", k) && !ex.input("v", v) &&
                    !ex.input("mask", repeat == 0 ? dense : row), "SDPA input");
            ncnn::Mat out;
            require(!ex.extract("out", out), "SDPA execution");
            require(out.w == width && out.h == queries && out.c == heads &&
                    out.elemsize == 4u && out.elempack == 1, "SDPA layout");
            for (int h = 0; h < heads; ++h)
                for (int i = 0; i < queries; ++i)
                {
                    const int visible = keys - 4 - shift - (mask_heads == 1 ? 0 : h);
                    std::vector<double> probabilities(visible);
                    double total = 0.;
                    if (!repeat && precision == "fp32")
                        for (int j = 0; j < visible; ++j)
                        {
                            double dot = 0.;
                            for (int x = 0; x < width; ++x)
                                dot += double(q.channel(h).row(i)[x]) * k.channel(0).row(j)[x];
                            probabilities[j] = std::exp(dot / std::sqrt(double(width)));
                            total += probabilities[j];
                        }
                    for (int d = 0; d < width; ++d)
                    {
                        const float actual = out.channel(h).row(i)[d];
                        require(std::isfinite(actual), "Nonfinite SDPA result");
                        if (repeat) require(actual == baseline.channel(h).row(i)[d], "Row mask differs from dense mask");
                        // Independent FP64 attention, with the mask represented
                        // by an exact visible-key count instead of a tensor.
                        if (!repeat && precision == "fp32")
                        {
                            double sum = 0.;
                            for (int j = 0; j < visible; ++j)
                                sum += probabilities[j] * v.channel(0).row(j)[d];
                            require(std::abs(actual - sum / total) < 3e-6, "FP64 reference differs");
                        }
                    }
                }
            if (!repeat) baseline = out;
        }
    }
    std::cout << "queries=" << queries << " keys=" << keys << " mask_heads=" << mask_heads
              << " width=" << width << " precision=" << precision << " dense/row/repeated=exact\n";
}
}

int main(int argc, char** argv)
{
    const bool vulkan = argc > 1 && std::string(argv[1]) == "vulkan";
    const std::string precision = argc > 2 ? argv[2] : "fp32";
#if NCNN_VULKAN
    if (vulkan)
    {
        ncnn::create_gpu_instance();
        if (!ncnn::get_gpu_count()) { ncnn::destroy_gpu_instance(); return 77; }
        const auto& info = ncnn::get_gpu_device()->info;
        if ((precision == "bf16" && !info.support_bf16_storage()) ||
            (precision == "fp16" && !info.support_fp16_storage()))
        { ncnn::destroy_gpu_instance(); return 77; }
    }
#else
    if (vulkan) return 77;
#endif
    int rc = 0;
    try
    {
        for (int heads : {1, 2})
        {
            run_case(vulkan, precision, 3, 17, heads);
            run_case(vulkan, precision, 257, 65, heads);
            // 124 is not divisible by eight: exercise the low-storage cross
            // fallback (including FP16 cooperative matrices), rather than Flash.
            run_case(vulkan, precision, 3, 17, heads, 124);
        }
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; rc = 1; }
#if NCNN_VULKAN
    if (vulkan) ncnn::destroy_gpu_instance();
#endif
    return rc;
}

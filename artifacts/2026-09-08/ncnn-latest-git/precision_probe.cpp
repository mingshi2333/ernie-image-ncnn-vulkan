// Standalone old/new ncnn diagnostic. All reduction inputs are exactly
// representable in FP32, FP16 and BF16; the oracle accumulates in double.
#include "net.h"
#include "gpu.h"
#include "command.h"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

static void check(int result) { if (result) throw std::runtime_error("ncnn return " + std::to_string(result)); }
static void number(double v) { if (std::isfinite(v)) std::cout << v; else std::cout << "null"; }

int main()
{
    std::cout << std::setprecision(17);
    int failures = 0;
    // Off-midpoint values distinguish round-to-nearest from truncation without
    // imposing an unadvertised halfway or subnormal contract on the helpers.
    struct Conversion { float input; unsigned short half, bf; };
    const Conversion cases[] = {{0.f,0x0000,0x0000}, {-0.f,0x8000,0x8000},
        {1.f,0x3c00,0x3f80}, {-1.f,0xbc00,0xbf80},
        {1.00075f,0x3c01,0x3f80}, {-1.00075f,0xbc01,0xbf80},
        {1.006f,0x3c06,0x3f81}, {-1.006f,0xbc06,0xbf81},
        {1.9998f,0x4000,0x4000}, {-1.9998f,0xc000,0xc000}};
    for (const auto& c : cases)
    {
        auto half = ncnn::float32_to_float16(c.input);
        auto bf = ncnn::float32_to_bfloat16(c.input);
        bool passed = half == c.half && bf == c.bf;
        failures += !passed;
        std::cout << "{\"kind\":\"conversion\",\"input\":" << c.input
                  << ",\"fp16_bits\":" << half << ",\"bf16_bits\":" << bf
                  << ",\"expected_fp16_bits\":" << c.half << ",\"expected_bf16_bits\":" << c.bf
                  << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
    }
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() < 1) { ncnn::destroy_gpu_instance(); return 77; }
    try
    {
        const auto* dev = ncnn::get_gpu_device(0);
        for (const std::string precision : {"fp32", "fp16", "bf16"})
        {
            if ((precision == "fp16" && !dev->info.support_fp16_storage()) ||
                (precision == "bf16" && !dev->info.support_bf16_storage()))
                throw std::runtime_error("requested storage unavailable: " + precision);
            for (const std::string pattern : {"constant256", "cancellation", "dyadic", "bf16_partial_cancellation"})
            {
                ncnn::VkBlobAllocator blobs(dev);
                ncnn::VkStagingAllocator staging(dev);
                ncnn::Option opt;
                opt.num_threads = 2;
                opt.use_vulkan_compute = true;
                opt.use_packing_layout = false;
                opt.use_fp16_packed = opt.use_bf16_packed = opt.use_fp16_arithmetic = false;
                opt.use_fp16_storage = precision == "fp16";
                opt.use_bf16_storage = precision == "bf16";
                opt.blob_vkallocator = opt.workspace_vkallocator = &blobs;
                opt.staging_vkallocator = &staging;
                auto layer = std::unique_ptr<ncnn::Layer>(ncnn::create_layer_vulkan("Reduction"));
                if (!layer || !layer->support_vulkan) throw std::runtime_error("no Vulkan Reduction");
                layer->vkdev = dev;
                ncnn::ParamDict pd;
                pd.set(0, 3); // mean
                pd.set(1, 1); // reduce all
                pd.set(2, 1.f);
                pd.set(4, 0);
                check(layer->load_param(pd));
                check(layer->create_pipeline(opt));
                {
                    constexpr int count = 4096;
                    ncnn::Mat input(count), output;
                    uint32_t rng = 7767517;
                    double expected = 0;
                    for (int i = 0; i < count; ++i)
                    {
                        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                        const int k = int(rng % 511) - 255;
                        float v = pattern == "constant256" ? 256.f : float(k) * .125f;
                        if (pattern == "cancellation") v *= (i < 2048 ? 16.f : .0625f);
                        if (pattern == "bf16_partial_cancellation") v = i == 0 ? 256.f : i == 1 ? 1.f : i == 32 ? -256.f : 0.f;
                        input[i] = v;
                        if (ncnn::float16_to_float32(ncnn::float32_to_float16(v)) != v ||
                            ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(v)) != v)
                            throw std::runtime_error("input rounding would confound reduction");
                        expected += v;
                    }
                    expected /= count;
                    double rounded = expected;
                    if (precision == "fp16") rounded = ncnn::float16_to_float32(ncnn::float32_to_float16(float(expected)));
                    if (precision == "bf16") rounded = ncnn::bfloat16_to_float32(ncnn::float32_to_bfloat16(float(expected)));
                    ncnn::VkMat uploaded, gi, go;
                    ncnn::VkCompute command(dev);
                    command.record_upload(input, uploaded, opt);
                    // Upload packs vectors automatically; direct Reduction
                    // requires pack1 (Net normally inserts this conversion).
                    dev->convert_packing(uploaded, gi, 1, command, opt);
                    check(layer->forward(gi, go, command, opt));
                    command.record_download(go, output, opt);
                    check(command.submit_and_wait());
                    if (output.w * output.h * output.d * output.c * output.elempack != 1 || output.elembits() != 32)
                        throw std::runtime_error("invalid download: dims=" + std::to_string(output.dims) + " w=" + std::to_string(output.w) + " h=" + std::to_string(output.h) + " d=" + std::to_string(output.d) + " c=" + std::to_string(output.c) + " bits=" + std::to_string(output.elembits()) + " pack=" + std::to_string(output.elempack) + " go=" + std::to_string(go.dims) + ":" + std::to_string(go.w));
                    const double actual = output[0];
                    const bool passed = std::isfinite(actual) && actual == rounded;
                    failures += !passed;
                    std::cout << "{\"kind\":\"reduction\",\"precision\":\"" << precision
                              << "\",\"pattern\":\"" << pattern << "\",\"elements\":" << count
                              << ",\"vulkan_direct\":true,\"storage_bits\":" << gi.elembits()
                              << ",\"oracle\":" << expected << ",\"rounded_oracle\":" << rounded << ",\"actual\":";
                    number(actual); std::cout << ",\"abs_error\":"; number(std::abs(actual-expected));
                    std::cout << ",\"finite\":" << (std::isfinite(actual) ? "true" : "false")
                              << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
                }
                check(layer->destroy_pipeline(opt));
            }
        }
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; failures++; }
    ncnn::destroy_gpu_instance();
    return failures ? 1 : 0;
}

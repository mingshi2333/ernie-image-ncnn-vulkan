// SPDX-License-Identifier: MIT
#include "ernie_gelu.h"
#include "latent_ops.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
void check(int rc)
{
    if (rc)
        throw std::runtime_error("Runtime error " + std::to_string(rc));
}
constexpr const char *graph = "7767517\n5 6\n"
                              "Input a 0 1 a\nInput b 0 1 b\n"
                              "ErnieResidualAdd sum 2 1 a b s 0=0\n"
                              "Split split 1 2 s sum norm_input\n"
                              "RMSNorm norm 1 1 norm_input normalized 0=4096 1=1e-6 2=1\n";
void verify(const ncnn::Mat &sum, const ncnn::Mat &norm, const ncnn::Mat &a, const ncnn::Mat &b)
{
    if (sum.w != 4096 || sum.h != 4 || norm.w != 4096 || norm.h != 4 || sum.elempack != 1 ||
        norm.elempack != 1 || sum.elemsize != 4u || norm.elemsize != 4u)
        throw std::runtime_error("Residual or normalization layout differs");
    for (int y = 0; y < 4; ++y)
    {
        double squares = 0;
        for (int i = 0; i < 4096; ++i)
        {
            const float expected = a.row(y)[i] + b.row(y)[i];
            if (sum.row(y)[i] != expected)
                throw std::runtime_error("FP32 residual differs or overflows");
            squares += double(expected) * expected;
        }
        const double scale = 1. / std::sqrt(squares / 4096 + 1e-6);
        for (int i = 0; i < 4096; ++i)
            if (std::abs(norm.row(y)[i] - sum.row(y)[i] * scale) > .001)
                throw std::runtime_error("FP32 skip to FP16 normalized projection differs");
    }
}
} // namespace
int main(int argc, char **argv)
{
    const bool gpu = argc == 2 && std::string(argv[1]) == "vulkan";
    int status = 0;
    try
    {
        ncnn::Option option;
        option.num_threads = 4;
        option.use_vulkan_compute = gpu;
        option.use_fp16_storage = gpu;
        option.use_fp16_packed = option.use_fp16_arithmetic = option.use_bf16_storage =
            option.use_bf16_packed = false;
        ncnn::Mat a(4096, 4), b(4096, 4), weights(4096);
        weights.fill(1.f);
        for (int y = 0; y < 4; ++y)
            for (int i = 0; i < 4096; ++i)
            {
                a.row(y)[i] = (i % 2 ? -1.f : 1.f) * (70000.f + y * 128.f);
                b.row(y)[i] = float(i % 8) * 128.f;
            }
        ncnn::Mat sum, norm;
        {
            ncnn::Net net;
            net.opt = option;
#if NCNN_VULKAN
            if (gpu)
            {
                ncnn::create_gpu_instance();
                if (ncnn::get_gpu_count() < 1)
                    return 77;
                net.set_vulkan_device(ncnn::get_default_gpu_index());
                if (!net.vulkan_device()->info.support_fp16_storage())
                    return 77;
            }
#else
            if (gpu)
                return 77;
#endif
            check(ernie::register_layers(net));
            check(net.load_param_mem(graph));
            if (net.load_model(static_cast<const unsigned char *>(weights.data)) != 4096 * sizeof(float))
                throw std::runtime_error("Normalization weight stream differs");
            if (!gpu)
            {
                auto ex = net.create_extractor();
                check(ex.input("a", a));
                check(ex.input("b", b));
                check(ex.extract("sum", sum));
                check(ex.extract("normalized", norm));
            }
#if NCNN_VULKAN
            else
            {
                const auto *device = net.vulkan_device();
                ncnn::VkBlobAllocator blobs(device);
                ncnn::VkStagingAllocator staging(device);
                option.blob_vkallocator = option.workspace_vkallocator = &blobs;
                option.staging_vkallocator = &staging;
                auto high = option;
                high.use_fp16_storage = false;
                ncnn::VkMat ga, gb, gs, gn;
                ncnn::VkCompute command(device);
                command.record_upload(a, ga, high);
                command.record_upload(b, gb, option);
                auto ex = net.create_extractor();
                ex.set_blob_vkallocator(&blobs);
                ex.set_workspace_vkallocator(&blobs);
                ex.set_staging_vkallocator(&staging);
                check(ex.input("a", ga));
                check(ex.input("b", gb));
                check(ex.extract("sum", gs, command));
                check(ex.extract("normalized", gn, command));
                if (gs.elembits() != 32 || gn.elembits() != 16)
                    throw std::runtime_error("Mixed storage contract differs");
                auto download = option;
                download.use_packing_layout = false;
                command.record_download(gs, sum, download);
                command.record_download(gn, norm, download);
                check(command.submit_and_wait());
                ernie::VulkanLatentOps ops(device);
                for (float bad : {1.f, INFINITY, NAN})
                {
                    ncnn::Mat values(3, 5, 128);
                    values.fill(1.f);
                    values.channel(42)[7] = bad;
                    ncnn::VkMat uploaded, packed;
                    ncnn::VkCompute upload(device);
                    upload.record_upload(values, uploaded, high);
                    device->convert_packing(uploaded, packed, 1, 1, upload, high);
                    check(upload.submit_and_wait());
                    if (ops.finite_latent(packed, device, option) != bool(std::isfinite(bad)))
                        throw std::runtime_error("GPU finite check differs");
                }
            }
#endif
        }
        verify(sum, norm, a, b);
        std::cout << "FP32 residual above 65504, normalized projection and finite guards passed\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
#if NCNN_VULKAN
    if (gpu)
        ncnn::destroy_gpu_instance();
#endif
    return status;
}

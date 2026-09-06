// SPDX-License-Identifier: MIT
#include "vae.h"
#include "ernie_gelu.h"
#include <filesystem>
#include <stdexcept>
namespace ernie
{
namespace
{
void check(int rc, const char *action)
{
    if (rc)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(rc));
}
} // namespace
ncnn::Mat decode_vae(const std::string &directory, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &vae_backend, const std::string &vae_convolution)
{
    const std::filesystem::path root(directory);
    ncnn::Mat decoded;
    {
        ncnn::Net vae;
        vae.opt = cpu;
        vae.opt.use_winograd_convolution = false;
        vae.opt.use_sgemm_convolution = vae_convolution == "sgemm";
#if NCNN_VULKAN
        if (vae_backend == "vulkan")
        {
            if (ncnn::get_gpu_count() < 1)
                throw std::runtime_error("No Vulkan device for VAE");
            vae.opt.use_vulkan_compute = true;
            vae.set_vulkan_device(ncnn::get_default_gpu_index());
        }
#else
        if (vae_backend == "vulkan")
            throw std::runtime_error("Built without Vulkan VAE support");
#endif
        check(ernie::register_layers(vae), "Register layers");
        check(vae.load_param((root / "head.ncnn.param").string().c_str()), "Load VAE graph");
        check(vae.load_model((root / "head.ncnn.bin").string().c_str()), "Load VAE weights");
        if (vae_backend == "vulkan")
            for (const auto *layer : vae.layers())
                if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                    throw std::runtime_error("VAE contains a compute layer without Vulkan support");
        auto ex = vae.create_extractor();
        check(ex.input("in0", unpacked), "Input VAE latent");
        check(ex.extract("out0", decoded), "Decode VAE");
    }
    return decoded;
}
} // namespace ernie

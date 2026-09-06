// SPDX-License-Identifier: MIT
#include "vae.h"
#include "ernie_gelu.h"
#include <filesystem>
#include <chrono>
#include <memory>
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
ncnn::Mat decode_vae(const ComponentFiles &files, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &vae_backend, const std::string &vae_convolution, int gpu_index,
                     VaeStats *stats)
{
    using Clock=std::chrono::steady_clock;
    if(stats)stats->details.clear();
    ncnn::Mat decoded;
    {
        auto vae=std::make_unique<ncnn::Net>();
        vae->opt = cpu;
        vae->opt.use_winograd_convolution = false;
        vae->opt.use_sgemm_convolution = vae_convolution == "sgemm";
#if NCNN_VULKAN
        if (vae_backend == "vulkan")
        {
            if (ncnn::get_gpu_count() < 1)
                throw std::runtime_error("No Vulkan device for VAE");
            vae->opt.use_vulkan_compute = true;
            const int selected = gpu_index >= 0 ? gpu_index : ncnn::get_default_gpu_index();
            if (selected < 0 || selected >= ncnn::get_gpu_count())
                throw std::invalid_argument("Requested Vulkan GPU index is unavailable for VAE");
            vae->set_vulkan_device(selected);
        }
#else
        if (vae_backend == "vulkan")
            throw std::runtime_error("Built without Vulkan VAE support");
#endif
        auto begin=Clock::now();
        try { check(ernie::register_layers(*vae), "Register layers");load_component_param(*vae,files); }
        catch (...) { if(stats)stats->details.push_back({"net_setup_param","failed",std::chrono::duration<double>(Clock::now()-begin).count()});throw; }
        if(stats)stats->details.push_back({"net_setup_param","complete",std::chrono::duration<double>(Clock::now()-begin).count()});
        begin=Clock::now();
        try { load_component_model(*vae,files); }
        catch (...) { if(stats)stats->details.push_back({"model_load_composite","failed",std::chrono::duration<double>(Clock::now()-begin).count()});throw; }
        if(stats)stats->details.push_back({"model_load_composite","complete",std::chrono::duration<double>(Clock::now()-begin).count()});
        if (vae_backend == "vulkan")
            for (const auto *layer : vae->layers())
                if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                    throw std::runtime_error("VAE contains a compute layer without Vulkan support");
        begin=Clock::now();
        try { auto ex = vae->create_extractor();check(ex.input("in0", unpacked), "Input VAE latent");check(ex.extract("out0", decoded), "Decode VAE"); }
        catch (...) { if(stats)stats->details.push_back({"extract_compute_composite","failed",std::chrono::duration<double>(Clock::now()-begin).count()});throw; }
        if(stats)stats->details.push_back({"extract_compute_composite","complete",std::chrono::duration<double>(Clock::now()-begin).count()});
        begin=Clock::now();vae.reset();if(stats)stats->details.push_back({"net_destroy","complete",std::chrono::duration<double>(Clock::now()-begin).count()});
    }
    return decoded;
}
ncnn::Mat decode_vae(const std::string &directory, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &device, const std::string &convolution, int gpu_index, VaeStats *stats)
{
    return decode_vae(component_files(std::filesystem::path(directory), "head"), unpacked, cpu,
                      device, convolution, gpu_index, stats);
}
} // namespace ernie

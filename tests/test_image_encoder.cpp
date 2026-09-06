// SPDX-License-Identifier: MIT
#include "image_encoder.h"
#include <iostream>
#include <stdexcept>

template<class F> void rejects(F f)
{
    try { f(); } catch (const std::exception &) { return; }
    throw std::runtime_error("Invalid encoder request accepted");
}

int main()
{
    try
    {
        ncnn::Option cpu; cpu.num_threads = 2;
        ernie::ComponentFiles missing{"", ""};
        ernie::RgbImage rgb{32, 32, std::vector<uint8_t>(32 * 32 * 3)};
        rejects([&] { ernie::encode_vae(missing, rgb, cpu); });
        auto wrong = rgb; wrong.width = 48; wrong.pixels.resize(48 * 32 * 3);
        rejects([&] { ernie::encode_vae({"7767517\n0 0\n", "/missing"}, wrong, cpu); });
        auto unreviewed_large = rgb; unreviewed_large.width = 384; unreviewed_large.height = 512;
        unreviewed_large.pixels.resize(size_t(384) * 512 * 3);
        rejects([&] { ernie::encode_vae({"7767517\n0 0\n", "/missing"}, unreviewed_large, cpu); });
        auto candidate = rgb; candidate.width = candidate.height = 1024;
        candidate.pixels.resize(size_t(1024) * 1024 * 3);
        // Production remains closed until the separately recorded candidate is reviewed.
        rejects([&] { ernie::encode_vae({"7767517\n0 0\n", "/missing"}, candidate, cpu); });
        // The evidence-only entry is fixed to 1024 and must reject every other shape
        // before it attempts to load a graph or weights.
        rejects([&] { ernie::encode_vae_candidate_1024({"7767517\n0 0\n", "/missing"}, rgb, cpu); });
        auto short_rgb = rgb; short_rgb.pixels.pop_back();
        rejects([&] { ernie::encode_vae({"7767517\n0 0\n", "/missing"}, short_rgb, cpu); });
        auto vulkan = cpu; vulkan.use_vulkan_compute = true;
        rejects([&] { ernie::encode_vae({"7767517\n0 0\n", "/missing"}, rgb, vulkan); });
        auto sgemm = cpu; sgemm.use_sgemm_convolution = true;
        rejects([&] { ernie::encode_vae({"7767517\n0 0\n", "/missing"}, rgb, sgemm); });
        std::cout << "VAE encoder request boundary contracts passed\n";
    }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

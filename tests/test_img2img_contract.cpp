// SPDX-License-Identifier: MIT
#include "denoiser.h"
#include "img2img.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
void require(bool condition, const char *reason)
{
    if (!condition) throw std::runtime_error(reason);
}
template <class F> void rejects(F action)
{
    try { action(); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("Invalid img2img request accepted");
}
void equal(const ncnn::Mat &a, const ncnn::Mat &b)
{
    require(a.w == b.w && a.h == b.h && a.c == b.c, "Latent shape differs");
    for (int c = 0; c < a.c; ++c)
        for (int i = 0; i < a.w * a.h; ++i)
            require(a.channel(c)[i] == b.channel(c)[i], "Endpoint latent differs");
}
}

int main()
{
    try
    {
        const auto schedule = ernie::FlowSchedule::turbo(8);
        ncnn::Mat encoded(2, 3, 128), noise(2, 3, 128);
        encoded.fill(2.f); noise.fill(-1.f);
        auto zero = ernie::make_img2img_start(encoded, ncnn::Mat(), schedule, 0.f, 1);
        require(zero.start_step == 8 && zero.denoise_steps == 0 && zero.sigma == 0.f, "Strength zero schedule differs");
        equal(zero.latent, encoded);
        zero.latent.channel(0)[0] = 123.f;
        require(encoded.channel(0)[0] == 2.f, "Reconstruction latent aliases input");
        auto one = ernie::make_img2img_start(encoded, noise, schedule, 1.f, 1);
        require(one.start_step == 0 && one.denoise_steps == 8 && one.sigma == 1.f, "Strength one schedule differs");
        equal(one.latent, noise);
        one.latent.channel(0)[0] = 123.f;
        require(noise.channel(0)[0] == -1.f, "Noise endpoint aliases input");
        for (const auto fraction : { .25f, .5f, .75f })
        {
            const int count = int(fraction * 8);
            const auto start = ernie::make_img2img_start(encoded, noise, schedule, fraction, 1);
            require(start.denoise_steps == count && start.start_step == 8 - count, "Strength step selection differs");
            const float expected = -start.sigma + (1.f - start.sigma) * 2.f;
            for (int c = 0; c < 128; ++c)
                for (int i = 0; i < 6; ++i)
                    require(start.latent.channel(c)[i] == expected, "Noise interpolation differs");
        }
        require(ernie::make_img2img_start(encoded, noise, schedule, .3125f, 1).denoise_steps == 3,
                "Half-step uses bankers rounding");
        require(ernie::make_img2img_start(encoded, noise, schedule, std::numeric_limits<float>::min(), 1).denoise_steps == 1,
                "Positive strength skipped all steps");
        for (const float bad : {-1.f, 1.01f, INFINITY, NAN})
            rejects([&] { ernie::make_img2img_start(encoded, noise, schedule, bad); });
        rejects([&] { ernie::make_img2img_start(encoded, noise, schedule, .5f, 0); });
        rejects([&] { ernie::make_img2img_start(encoded, ncnn::Mat(1, 1, 128), schedule, .5f); });
        auto malformed = schedule;
        malformed.sigmas[4] = NAN;
        rejects([&] { ernie::make_img2img_start(encoded, noise, malformed, .5f); });
        malformed = schedule; malformed.timesteps.pop_back();
        rejects([&] { ernie::make_img2img_start(encoded, noise, malformed, .5f); });
        noise.channel(1)[0] = NAN;
        rejects([&] { ernie::make_img2img_start(encoded, noise, schedule, .5f); });
        noise.channel(1)[0] = -1.f;

        // No file in this model exists. A completed schedule must not load DiT,
        // emit a prediction or retain a stale stats entry from a prior request.
        const ernie::DenoiseModel absent{{"unloaded input graph", "/nonexistent/input.bin"},
                                         {"unloaded output graph", "/nonexistent/output.bin"},
                                         {{"unloaded block graph", "/nonexistent/block.bin"}}};
        const std::vector<ncnn::Mat> constants(4);
        ncnn::Option cpu; cpu.use_vulkan_compute = false;
        std::vector<ernie::DenoiseStepStats> stats(1);
        bool called = false;
        const auto result = ernie::denoise(absent, encoded, constants, 8, cpu, stats,
                                           [&](size_t, const ncnn::Mat &, const ncnn::Mat &) { called = true; }, 8);
        equal(result, encoded);
        require(stats.empty() && !called, "Reconstruction executed a denoising step");
        rejects([&] { ernie::denoise(absent, encoded, constants, 8, cpu, stats, {}, -1); });
        rejects([&] { ernie::denoise(absent, encoded, constants, 8, cpu, stats, {}, 9); });
        std::cout << "Img2img endpoints, interpolation, positive rounding, rejection and no-denoise reconstruction passed\n";
    }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

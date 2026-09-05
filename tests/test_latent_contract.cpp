// SPDX-License-Identifier: MIT
#include "latent_ops.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

template<class F> void rejects(F&& action)
{
    try { action(); }
    catch (const std::invalid_argument&) { return; }
    catch (const std::out_of_range&) { return; }
    throw std::runtime_error("Expected invalid request to fail");
}

int main()
{
    try
    {
        rejects([] { ernie::FlowSchedule::turbo(0); });
        rejects([] { ernie::FlowSchedule::turbo(1001); });
        auto schedule = ernie::FlowSchedule::turbo();
        rejects([&] { schedule.delta(8); });
        if (schedule.sigmas.front() != 1.f || schedule.sigmas.back() != 0.f)
            throw std::runtime_error("Schedule endpoints differ");
        for (size_t i = 0; i < 8; ++i)
            if (schedule.delta(i) >= 0.f) throw std::runtime_error("Schedule does not descend");
        ncnn::Mat input(3, 5, 128), prediction(3, 5, 128), bad(3, 5, 32), output, stats(128), variance(128);
        input.fill(1.f); prediction.fill(1.f); stats.fill(0.f); variance.fill(1.f);
        rejects([&] { ernie::euler_step(input, bad, -.1f, output); });
        rejects([&] { ernie::euler_step(input, prediction, 0.f, output); });
        rejects([&] { ernie::euler_step(input, prediction, NAN, output); });
        variance[42] = -1.f;
        rejects([&] { ernie::unpack_for_vae(input, stats, variance); });
        variance[42] = 1.f;
        // Alias-safe consume-and-replace is needed by the denoising loop.
        ernie::euler_step(input, prediction, -1.f, input);
        for (int c = 0; c < 128; ++c)
            for (int i = 0; i < 15; ++i)
                if (input.channel(c)[i] != 0.f) throw std::runtime_error("Aliased update corrupted output");
        std::cout << "Schedule, shape, invalid delta, BN statistics, and aliased update contracts passed\n";
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

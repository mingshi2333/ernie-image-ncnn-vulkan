// SPDX-License-Identifier: MIT
#include "img2img.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ernie
{
Img2ImgStart make_img2img_start(const ncnn::Mat &encoded, const ncnn::Mat &saved_noise,
                              const FlowSchedule &schedule, float strength, int threads)
{
    if (!std::isfinite(strength) || strength < 0.f || strength > 1.f || threads < 1 || threads > 256)
        throw std::invalid_argument("Img2img requires strength in [0,1] and threads in [1,256]");
    const size_t count = schedule.timesteps.size();
    if (count < 1 || count > 1000 || schedule.sigmas.size() != count + 1 ||
        schedule.sigmas.front() != 1.f || schedule.sigmas.back() != 0.f)
        throw std::invalid_argument("Invalid img2img flow schedule");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(schedule.sigmas[i]) || !std::isfinite(schedule.timesteps[i]) ||
            schedule.sigmas[i] <= schedule.sigmas[i + 1] ||
            schedule.sigmas[i] < 0.f || schedule.sigmas[i] > 1.f)
            throw std::invalid_argument("Img2img flow schedule must descend from one to zero");
    if (!finite_latent(encoded))
        throw std::invalid_argument("Encoded img2img latent must be finite");
    Img2ImgStart result;
    result.start_step = int(count);
    if (strength == 0.f)
    {
        result.latent = encoded.clone();
        if (result.latent.empty()) throw std::bad_alloc();
        return result;
    }
    if (!finite_latent(saved_noise) || saved_noise.w != encoded.w || saved_noise.h != encoded.h)
        throw std::invalid_argument("Img2img noise must be finite and match the encoded shape");
    const float scaled = float(count) * strength;
    result.denoise_steps = std::min(int(count), std::max(1, int(std::floor(scaled + .5f))));
    result.start_step = int(count) - result.denoise_steps;
    result.sigma = schedule.sigmas[size_t(result.start_step)];
    if (result.sigma == 1.f)
    {
        result.latent = saved_noise.clone();
        if (result.latent.empty()) throw std::bad_alloc();
        return result;
    }
    result.latent.create(encoded.w, encoded.h, 128);
    if (result.latent.empty()) throw std::bad_alloc();
    const float retained = 1.f - result.sigma;
    #pragma omp parallel for num_threads(threads)
    for (int c = 0; c < 128; ++c)
    {
        const float *image = encoded.channel(c), *noise = saved_noise.channel(c);
        float *out = result.latent.channel(c);
        for (int i = 0; i < encoded.w * encoded.h; ++i)
        {
            const float weighted_noise = result.sigma * noise[i];
            const float weighted_image = retained * image[i];
            out[i] = weighted_noise + weighted_image;
        }
    }
    if (!finite_latent(result.latent))
        throw std::runtime_error("Non-finite img2img initial latent");
    return result;
}
} // namespace ernie

// SPDX-License-Identifier: MIT
#pragma once
#include "latent_ops.h"

namespace ernie
{
struct Img2ImgStart
{
    ncnn::Mat latent; // Independent pack1 FP32 storage; never aliases the inputs.
    int start_step = 0;
    int denoise_steps = 0;
    float sigma = 0.f;
};

// The encoder input is the already patchified, BN-normalized mean posterior.
// Strength zero returns its reconstruction latent without consuming noise.
// Positive strengths use round-half-up step selection from the pinned peer.
Img2ImgStart make_img2img_start(const ncnn::Mat &encoded, const ncnn::Mat &saved_noise,
                              const FlowSchedule &schedule, float strength, int threads = 4);
} // namespace ernie

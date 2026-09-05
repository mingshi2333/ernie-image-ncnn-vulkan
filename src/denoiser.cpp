// SPDX-License-Identifier: MIT
#include "denoiser.h"
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ernie
{
namespace
{
using Clock = std::chrono::steady_clock;
template <class T> void check_request(const DenoiseModel &model, const T &initial, size_t constants)
{
    if (model.input_head.empty() || model.output_head.empty() || model.blocks.empty() ||
        model.blocks.size() > 36 || constants != 4 || initial.empty() || initial.dims != 3 ||
        initial.elempack != 1 || initial.elemsize != 4u || initial.c != 128 || initial.w < 1 ||
        initial.h < 1 || initial.w > 256 || initial.h > 256)
        throw std::invalid_argument("Require heads, 1..36 blocks, four constants and a pack1 FP32 latent");
}
} // namespace
ncnn::Mat timestep_features(float timestep)
{
    if (!std::isfinite(timestep) || timestep < 0.f || timestep > 1000.f)
        throw std::invalid_argument("Timestep must be finite and in [0,1000]");
    ncnn::Mat out(4096);
    if (out.empty())
        throw std::bad_alloc();
    const float factor = -float(std::log(10000.0));
    for (int i = 0; i < 2048; ++i)
    {
        const float product = factor * float(i);
        const float exponent = product / 2048.f;
        const float phase = timestep * std::exp(exponent);
        out[i] = std::sin(phase);
        out[2048 + i] = std::cos(phase);
    }
    return out;
}

ncnn::Mat denoise(const DenoiseModel &model, const ncnn::Mat &initial,
                  const std::vector<ncnn::Mat> &constants, int steps, const ncnn::Option &option,
                  std::vector<DenoiseStepStats> &stats, const CpuStepObserver &observer)
{
    check_request(model, initial, constants.size());
    if (option.use_vulkan_compute)
        throw std::invalid_argument("CPU denoiser requires CPU options");
    const auto schedule = FlowSchedule::turbo(steps);
    stats.clear();
    ncnn::Mat sample = initial;
    if (!finite_latent(sample))
        throw std::invalid_argument("Initial latent is not finite");
    for (int i = 0; i < steps; ++i)
    {
        const auto start = Clock::now();
        DenoiseStepStats step;
        step.timestep = schedule.timesteps[i];
        step.delta = schedule.delta(i);
        const auto features = timestep_features(step.timestep);
        const std::vector<ncnn::Mat> inputs{sample,       constants[0], features,
                                            constants[1], constants[2], constants[3]};
        const auto prediction =
            run_dit(model.input_head, model.blocks, model.output_head, inputs, option, step.dit);
        ncnn::Mat next;
        euler_step(sample, prediction, step.delta, next, option.num_threads);
        sample = next;
        if (!finite_latent(sample))
            throw std::runtime_error("Non-finite latent after denoise step " + std::to_string(i + 1));
        step.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        stats.push_back(step);
        if (observer)
            observer(i, prediction, sample);
    }
    return sample;
}

#if NCNN_VULKAN
ncnn::VkMat denoise(const DenoiseModel &model, const ncnn::VkMat &initial,
                    const std::vector<ncnn::VkMat> &constants, int steps, const ncnn::VulkanDevice *device,
                    const ncnn::Option &option, std::vector<DenoiseStepStats> &stats,
                    const VulkanStepObserver &observer)
{
    check_request(model, initial, constants.size());
    if (!device || !option.use_vulkan_compute || !option.blob_vkallocator || !option.staging_vkallocator)
        throw std::invalid_argument("Vulkan denoiser requires a device and session allocators");
    const auto schedule = FlowSchedule::turbo(steps);
    stats.clear();
    VulkanLatentOps latent_ops(device);
    ncnn::VkMat sample = initial;
    if (!latent_ops.finite_latent(sample, device, option))
        throw std::invalid_argument("Initial latent is not finite");
    for (int i = 0; i < steps; ++i)
    {
        const auto start = Clock::now();
        DenoiseStepStats step;
        step.timestep = schedule.timesteps[i];
        step.delta = schedule.delta(i);
        const auto features = timestep_features(step.timestep);
        ncnn::VkMat model_input, time_input;
        {
            ncnn::VkCompute command(device);
            const int storage = option.use_bf16_storage ? 5 : option.use_fp16_storage ? 2 : 1;
            device->convert_packing(sample, model_input, 4, storage, command, option);
            command.record_upload(features, time_input, option);
            if (command.submit_and_wait())
                throw std::runtime_error("Prepare DiT step failed");
        }
        const std::vector<ncnn::VkMat> inputs{model_input,  constants[0], time_input,
                                              constants[1], constants[2], constants[3]};
        const auto output =
            run_dit(model.input_head, model.blocks, model.output_head, inputs, device, option, step.dit);
        ncnn::VkMat prediction, next;
        {
            ncnn::VkCompute command(device);
            device->convert_packing(output, prediction, 1, 1, command, option);
            latent_ops.record_euler(sample, prediction, step.delta, next, command, option);
            if (command.submit_and_wait())
                throw std::runtime_error("Euler step failed");
        }
        sample = next;
        if (!latent_ops.finite_latent(sample, device, option))
            throw std::runtime_error("Non-finite latent after denoise step " + std::to_string(i + 1));
        step.elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        stats.push_back(step);
        if (observer)
            observer(i, prediction, sample);
    }
    return sample;
}
#endif
} // namespace ernie

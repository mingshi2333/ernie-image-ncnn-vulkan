// SPDX-License-Identifier: MIT
#include "latent_ops.h"
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ernie {
namespace {
template<class T> void check_latent(const T& value)
{
    if (value.empty() || value.dims != 3 || value.elempack != 1 || value.elemsize != 4u
        || value.c != 128 || value.w < 1 || value.h < 1 || value.w > 256 || value.h > 256)
        throw std::invalid_argument("Expected pack1 FP32 [128,H,W] latent with 1 <= H,W <= 256");
}
template<class T> void check_stats(const T& value)
{
    if (value.dims != 1 || value.w != 128 || value.elemsize != 4u || value.elempack != 1 || value.empty())
        throw std::invalid_argument("Expected pack1 FP32 [128] BN statistics");
}
template<class T> void check_pair(const T& a, const T& b, float delta)
{
    check_latent(a);
    check_latent(b);
    if (a.w != b.w || a.h != b.h || !std::isfinite(delta) || delta >= 0.f || delta < -1.f)
        throw std::invalid_argument("Incompatible Euler shapes or delta outside [-1,0)");
}
} // namespace

FlowSchedule FlowSchedule::turbo(int steps)
{
    if (steps < 1 || steps > 1000) throw std::invalid_argument("Steps must be in [1,1000]");
    FlowSchedule result;
    result.sigmas.resize(steps + 1);
    result.timesteps.resize(steps);
    const float increment = -1.f / steps;
    for (int i = 0; i < steps; ++i)
    {
        // Match PyTorch's CPU FP32 linspace: evaluate from the nearer endpoint.
        // FMA here is intentional; shifting below follows separate NumPy ops.
        const float raw = i < (steps + 1) / 2 ? std::fma(increment, float(i), 1.f)
                                                : -increment * float(steps - i);
        const float scaled = 4.f * raw;
        const float denominator = 1.f + 3.f * raw;
        result.sigmas[i] = scaled / denominator;
        result.timesteps[i] = 1000.f * result.sigmas[i];
    }
    result.sigmas[steps] = 0.f;
    return result;
}

float FlowSchedule::delta(size_t step) const
{
    if (step >= timesteps.size() || sigmas.size() != timesteps.size() + 1)
        throw std::out_of_range("Euler schedule exhausted or malformed");
    return sigmas[step + 1] - sigmas[step];
}

bool finite_latent(const ncnn::Mat& value)
{
    check_latent(value);
    for (int c = 0; c < value.c; ++c)
    {
        const float* data = value.channel(c);
        for (int i = 0; i < value.w * value.h; ++i)
            if (!std::isfinite(data[i])) return false;
    }
    return true;
}

void euler_step(const ncnn::Mat& sample, const ncnn::Mat& prediction, float delta,
                ncnn::Mat& next, int threads)
{
    check_pair(sample, prediction, delta);
    if (threads < 1) throw std::invalid_argument("Thread count must be positive");
    ncnn::Mat result(sample.w, sample.h, sample.c);
    if (result.empty()) throw std::bad_alloc();
    #pragma omp parallel for num_threads(threads)
    for (int c = 0; c < sample.c; ++c)
    {
        const float* x = sample.channel(c);
        const float* v = prediction.channel(c);
        float* out = result.channel(c);
        for (int i = 0; i < sample.w * sample.h; ++i)
        {
            const float change = delta * v[i];
            out[i] = x[i] + change;
        }
    }
    next = result;
}

ncnn::Mat unpack_for_vae(const ncnn::Mat& packed, const ncnn::Mat& mean,
                         const ncnn::Mat& variance, int threads)
{
    check_latent(packed);
    check_stats(mean);
    check_stats(variance);
    if (threads < 1) throw std::invalid_argument("Thread count must be positive");
    float scales[128];
    for (int c = 0; c < 128; ++c)
    {
        if (!std::isfinite(mean[c]) || !std::isfinite(variance[c]) || variance[c] < 0.f)
            throw std::invalid_argument("Invalid BN statistics");
        scales[c] = std::sqrt(variance[c] + 1e-5f);
    }
    ncnn::Mat out(packed.w * 2, packed.h * 2, 32);
    if (out.empty()) throw std::bad_alloc();
    #pragma omp parallel for num_threads(threads)
    for (int c = 0; c < 128; ++c)
    {
        const float* values = packed.channel(c);
        float* target = out.channel(c / 4);
        const int dy = (c % 4) / 2, dx = c % 2;
        for (int y = 0; y < packed.h; ++y)
            for (int x = 0; x < packed.w; ++x)
            {
                const float scaled = values[y * packed.w + x] * scales[c];
                target[(2 * y + dy) * out.w + 2 * x + dx] = scaled + mean[c];
            }
    }
    return out;
}

#if NCNN_VULKAN
namespace {
constexpr const char* euler_shader = R"glsl(
#version 450
layout(binding=0) readonly buffer sample_buffer { float sample_data[]; };
layout(binding=1) readonly buffer prediction_buffer { float prediction_data[]; };
layout(binding=2) writeonly buffer output_buffer { float output_data[]; };
layout(push_constant) uniform parameter { uint plane; uint channels; uint sample_stride;
    uint prediction_stride; uint output_stride; float delta; } p;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.plane * p.channels) return;
    uint c = i / p.plane, offset = i % p.plane;
    precise float change = p.delta * prediction_data[c * p.prediction_stride + offset];
    precise float value = sample_data[c * p.sample_stride + offset] + change;
    output_data[c * p.output_stride + offset] = value;
}
)glsl";
constexpr const char* unpack_shader = R"glsl(
#version 450
layout(binding=0) readonly buffer input_buffer { float input_data[]; };
layout(binding=1) readonly buffer mean_buffer { float mean_data[]; };
layout(binding=2) readonly buffer variance_buffer { float variance_data[]; };
layout(binding=3) writeonly buffer output_buffer { float output_data[]; };
layout(push_constant) uniform parameter { uint w; uint h; uint input_stride; uint output_stride; } p;
void main()
{
    uint i = gl_GlobalInvocationID.x, plane = p.w * p.h;
    if (i >= 128 * plane) return;
    uint c = i / plane, offset = i % plane;
    uint y = offset / p.w, x = offset % p.w;
    precise float variance = variance_data[c] + 1e-5;
    precise float scaled = input_data[c * p.input_stride + offset] * sqrt(variance);
    precise float value = scaled + mean_data[c];
    uint target = (2 * y + (c % 4) / 2) * (2 * p.w) + 2 * x + c % 2;
    output_data[(c / 4) * p.output_stride + target] = value;
}
)glsl";

constexpr const char* finite_shader = R"glsl(
#version 450
layout(binding=0) readonly buffer values_buffer { float values[]; };
layout(binding=1) writeonly buffer flags_buffer { float flags[]; };
layout(push_constant) uniform parameter { uint plane; uint stride; } p;
void main()
{
    uint c = gl_GlobalInvocationID.x;
    if (c >= 128) return;
    float ok = 1.f;
    for (uint i = 0; i < p.plane; ++i)
    {
        float v = values[c * p.stride + i];
        if (isnan(v) || isinf(v)) { ok = 0.f; break; }
    }
    flags[c] = ok;
}
)glsl";

std::unique_ptr<ncnn::Pipeline> make_pipeline(const char* source, const ncnn::VulkanDevice* device)
{
    ncnn::Option option;
    option.use_fp16_storage = option.use_fp16_packed = option.use_fp16_arithmetic = false;
    option.use_bf16_storage = option.use_bf16_packed = false;
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(source, option, spirv)) throw std::runtime_error("Latent shader compilation failed");
    auto pipeline = std::make_unique<ncnn::Pipeline>(device);
    pipeline->set_local_size_xyz(128, 1, 1);
    if (pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), {}))
        throw std::runtime_error("Latent pipeline creation failed");
    return pipeline;
}

void check_options(const ncnn::Option& option)
{
    if (!option.blob_vkallocator) throw std::invalid_argument("A caller-owned Vulkan allocator is required");
}
} // namespace

VulkanLatentOps::VulkanLatentOps(const ncnn::VulkanDevice* device)
{
    if (!device) throw std::invalid_argument("A Vulkan device is required");
    auto euler = make_pipeline(euler_shader, device);
    auto unpack = make_pipeline(unpack_shader, device);
    auto finite = make_pipeline(finite_shader, device);
    euler_ = euler.release();
    unpack_ = unpack.release();
    finite_ = finite.release();
}

VulkanLatentOps::~VulkanLatentOps() { delete euler_; delete unpack_; delete finite_; }

bool VulkanLatentOps::finite_latent(const ncnn::VkMat& value, const ncnn::VulkanDevice* device,
                                   const ncnn::Option& option) const
{
    check_latent(value);
    check_options(option);
    ncnn::VkMat flags(128, size_t(4), 1, option.blob_vkallocator);
    if (flags.empty()) throw std::bad_alloc();
    ncnn::VkCompute command(device);
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = value.w * value.h;
    constants[1].u32 = value.cstep;
    command.record_pipeline(finite_, {value, flags}, constants, flags);
    ncnn::Option high = option;
    high.use_fp16_storage = high.use_fp16_packed = high.use_fp16_arithmetic = false;
    high.use_bf16_storage = high.use_bf16_packed = high.use_packing_layout = false;
    ncnn::Mat result;
    command.record_download(flags, result, high);
    if (command.submit_and_wait()) throw std::runtime_error("Latent finite check failed");
    for (int i = 0; i < 128; ++i) if (result[i] != 1.f) return false;
    return true;
}

void VulkanLatentOps::record_euler(const ncnn::VkMat& sample, const ncnn::VkMat& prediction,
                                  float delta, ncnn::VkMat& next, ncnn::VkCompute& command,
                                  const ncnn::Option& option) const
{
    check_pair(sample, prediction, delta);
    check_options(option);
    ncnn::VkMat result(sample.w, sample.h, sample.c, size_t(4), 1, option.blob_vkallocator);
    if (result.empty()) throw std::bad_alloc();
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].u32 = sample.w * sample.h;
    constants[1].u32 = sample.c;
    constants[2].u32 = sample.cstep;
    constants[3].u32 = prediction.cstep;
    constants[4].u32 = result.cstep;
    constants[5].f = delta;
    ncnn::VkMat dispatch;
    dispatch.w = sample.w * sample.h * sample.c;
    dispatch.h = dispatch.c = 1;
    command.record_pipeline(euler_, {sample, prediction, result}, constants, dispatch);
    next = result;
}

void VulkanLatentOps::record_unpack(const ncnn::VkMat& packed, const ncnn::VkMat& mean,
                                   const ncnn::VkMat& variance, ncnn::VkMat& unpacked,
                                   ncnn::VkCompute& command, const ncnn::Option& option) const
{
    check_latent(packed);
    check_stats(mean);
    check_stats(variance);
    check_options(option);
    ncnn::VkMat result(packed.w * 2, packed.h * 2, 32, size_t(4), 1, option.blob_vkallocator);
    if (result.empty()) throw std::bad_alloc();
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].u32 = packed.w;
    constants[1].u32 = packed.h;
    constants[2].u32 = packed.cstep;
    constants[3].u32 = result.cstep;
    ncnn::VkMat dispatch;
    dispatch.w = packed.w * packed.h * packed.c;
    dispatch.h = dispatch.c = 1;
    command.record_pipeline(unpack_, {packed, mean, variance, result}, constants, dispatch);
    unpacked = result;
}
#endif
} // namespace ernie

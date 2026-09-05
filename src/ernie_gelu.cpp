// SPDX-License-Identifier: MIT
#include "ernie_gelu.h"
#include <cmath>
#include <vector>

namespace ernie {
namespace {
#if NCNN_VULKAN
// Abramowitz & Stegun, 7.1.26. Evaluate erfc(|x|) directly to retain the
// negative GELU tail. All arithmetic is FP32, including low-precision storage.
// This is a numerical erf approximation, not bit-exact libm or tanh GELU.
constexpr const char* shader = R"glsl(
#version 450
layout(binding = 0) buffer data_buffer { sfpvec4 values[]; };
layout(push_constant) uniform parameter { uint count; } p;
void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    vec4 x = vec4(buffer_ld4(values, i));
    vec4 z = abs(x) * 0.7071067811865475;
    vec4 t = 1.0 / (1.0 + 0.3275911 * z);
    vec4 tail = (((((1.061405429 * t - 1.453152027) * t)
                   + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
    vec4 probability = mix(0.5 * tail, 1.0 - 0.5 * tail, greaterThanEqual(x, vec4(0.0)));
    buffer_st4(values, i, afpvec4(x * probability));
}
)glsl";
#endif

class ErnieGELU final : public ncnn::Layer
{
public:
    ErnieGELU()
    {
        one_blob_only = true;
        support_inplace = true;
        support_packing = true;
#if NCNN_VULKAN
        support_vulkan = true;
        support_vulkan_packing = true;
#endif
    }

    int forward_inplace(ncnn::Mat& blob, const ncnn::Option& opt) const override
    {
        if (blob.elemsize / blob.elempack != sizeof(float)) return -1;
        const size_t count = size_t(blob.w) * blob.h * blob.d * blob.elempack;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < blob.c; ++c)
        {
            float* values = blob.channel(c);
            for (size_t i = 0; i < count; ++i)
                values[i] = 0.5f * values[i] * std::erfc(-0.7071067811865475f * values[i]);
        }
        return 0;
    }

#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& opt) override
    {
        if (!opt.use_vulkan_compute) return 0;
        std::vector<uint32_t> spirv;
        ncnn::Option shader_opt = opt;
        shader_opt.use_fp16_arithmetic = false;
        int rc = ncnn::compile_spirv_module(shader, shader_opt, spirv);
        if (rc) return rc;
        pipeline = new ncnn::Pipeline(vkdev);
        pipeline->set_local_size_xyz(64, 1, 1);
        return pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t), {});
    }

    int destroy_pipeline(const ncnn::Option&) override
    {
        delete pipeline;
        pipeline = nullptr;
        return 0;
    }

    int forward_inplace(ncnn::VkMat& blob, ncnn::VkCompute& cmd, const ncnn::Option&) const override
    {
        // ERNIE FFN widths are divisible by four. Reject other shapes instead
        // of silently leaving the tail of an arbitrary tensor untouched.
        const size_t elements = blob.total() * blob.elempack;
        if (!pipeline || elements % 4) return -1;
        std::vector<ncnn::VkMat> bindings{blob};
        std::vector<ncnn::vk_constant_type> constants(1);
        constants[0].u32 = elements / 4;
        ncnn::VkMat dispatcher;
        dispatcher.w = elements / 4;
        dispatcher.h = dispatcher.c = 1;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        return 0;
    }

private:
    ncnn::Pipeline* pipeline = nullptr;
#endif
};

DEFINE_LAYER_CREATOR(ErnieGELU)
} // namespace

int register_layers(ncnn::Net& net)
{
    return net.register_custom_layer("ErnieGELU", ErnieGELU_layer_creator);
}
} // namespace ernie

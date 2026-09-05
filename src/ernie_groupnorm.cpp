// SPDX-License-Identifier: MIT
#include "ernie_groupnorm.h"
#include "layer.h"
#include <cmath>

namespace ernie
{
namespace
{
// Large VAE planes accumulate millions of values per group. Use FP64 mean
// and centered variance reductions; activations and affine arithmetic stay FP32.
class ErnieGroupNorm final : public ncnn::Layer
{
  public:
    ErnieGroupNorm() { one_blob_only = support_inplace = true; }
    int load_param(const ncnn::ParamDict &pd) override
    {
        groups = pd.get(0, 1);
        channels = pd.get(1, 0);
        eps = pd.get(2, .001f);
        affine = pd.get(3, 1);
        return groups > 0 && channels > 0 && channels % groups == 0 && std::isfinite(eps) && eps > 0.f ? 0
                                                                                                       : -1;
    }
    int load_model(const ncnn::ModelBin &model) override
    {
        if (!affine)
            return 0;
        gamma = model.load(channels, 1);
        beta = model.load(channels, 1);
        return gamma.empty() || beta.empty() ? -100 : 0;
    }
    int forward_inplace(ncnn::Mat &value, const ncnn::Option &option) const override
    {
        if (value.empty() || value.elempack != 1 || value.elembits() != 32 ||
            (value.dims != 2 && value.dims != 3) || (value.dims == 2 ? value.h : value.c) != channels)
            return -1;
        const int plane = value.dims == 2 ? value.w : value.w * value.h;
        const size_t stride = value.dims == 2 ? value.w : value.cstep;
        const int per_group = channels / groups;
        float *data = value;
#pragma omp parallel for num_threads(option.num_threads)
        for (int g = 0; g < groups; ++g)
        {
            double total = 0, variance = 0;
            for (int c = g * per_group; c < (g + 1) * per_group; ++c)
                for (int i = 0; i < plane; ++i)
                    total += data[c * stride + i];
            const double mean = total / (double(plane) * per_group);
            for (int c = g * per_group; c < (g + 1) * per_group; ++c)
                for (int i = 0; i < plane; ++i)
                {
                    const double delta = data[c * stride + i] - mean;
                    variance += delta * delta;
                }
            const float inverse = float(1. / std::sqrt(variance / (double(plane) * per_group) + eps));
            for (int c = g * per_group; c < (g + 1) * per_group; ++c)
            {
                const float scale = inverse * (affine ? gamma[c] : 1.f);
                const float bias = (affine ? beta[c] : 0.f) - float(mean) * scale;
                float *channel = data + c * stride;
                for (int i = 0; i < plane; ++i)
                    channel[i] = channel[i] * scale + bias;
            }
        }
        return 0;
    }
    int groups = 0, channels = 0, affine = 1;
    float eps = 0;
    ncnn::Mat gamma, beta;
};
DEFINE_LAYER_CREATOR(ErnieGroupNorm)
} // namespace
int register_groupnorm(ncnn::Net &net)
{
    // Vulkan uses the pinned native implementation; do not create a hidden
    // CPU fallback in the explicit all-Vulkan VAE path.
    if (net.opt.use_vulkan_compute)
        return 0;
    return net.register_custom_layer("GroupNorm", ErnieGroupNorm_layer_creator);
}
} // namespace ernie

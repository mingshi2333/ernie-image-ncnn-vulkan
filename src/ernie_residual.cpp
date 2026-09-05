// SPDX-License-Identifier: MIT
#include "ernie_residual.h"
#include "layer.h"
#include <memory>

namespace ernie
{
namespace
{
ncnn::Option high(const ncnn::Option &option)
{
    auto result = option;
    result.use_fp16_storage = result.use_fp16_packed = result.use_fp16_arithmetic = false;
    result.use_bf16_storage = result.use_bf16_packed = false;
    return result;
}
// Residual sums can exceed 65504 with real text, even when their normalized
// projections fit FP16. Keep the skip path and addition in FP32 on the GPU.
class ErnieResidualAdd final : public ncnn::Layer
{
  public:
    ErnieResidualAdd() : cpu(ncnn::create_layer_cpu("BinaryOp"))
    {
        one_blob_only = false;
        support_packing = true;
#if NCNN_VULKAN
        support_vulkan = support_vulkan_packing = true;
        gpu.reset(ncnn::create_layer_vulkan("BinaryOp"));
#endif
    }
    int load_param(const ncnn::ParamDict &params) override
    {
        if (params.get(0, -1) != 0 || params.get(1, 0) != 0 || !cpu)
            return -1;
        int rc = cpu->load_param(params);
#if NCNN_VULKAN
        if (!rc)
            rc = gpu ? gpu->load_param(params) : -1;
#endif
        return rc;
    }
    int create_pipeline(const ncnn::Option &option) override
    {
#if NCNN_VULKAN
        if (option.use_vulkan_compute)
        {
            gpu->vkdev = vkdev;
            return gpu->create_pipeline(high(option));
        }
#endif
        return cpu->create_pipeline(high(option));
    }
    int destroy_pipeline(const ncnn::Option &option) override
    {
        int rc = cpu->destroy_pipeline(high(option));
#if NCNN_VULKAN
        if (gpu)
            gpu->destroy_pipeline(high(option));
#endif
        return rc;
    }
    int forward(const std::vector<ncnn::Mat> &inputs, std::vector<ncnn::Mat> &outputs,
                const ncnn::Option &option) const override
    {
        if (inputs.size() != 2 || inputs[0].elembits() != 32 || inputs[1].elembits() != 32)
            return -1;
        return cpu->forward(inputs, outputs, high(option));
    }
#if NCNN_VULKAN
    int forward(const std::vector<ncnn::VkMat> &inputs, std::vector<ncnn::VkMat> &outputs,
                ncnn::VkCompute &command, const ncnn::Option &option) const override
    {
        if (inputs.size() != 2)
            return -1;
        std::vector<ncnn::VkMat> promoted(2);
        for (int i = 0; i < 2; ++i)
        {
            if (inputs[i].elembits() == 32)
                promoted[i] = inputs[i];
            else if (inputs[i].elembits() == 16)
                vkdev->convert_packing(inputs[i], promoted[i], inputs[i].elempack, 1, command, option);
            else
                return -1;
            if (promoted[i].empty())
                return -100;
        }
        return gpu->forward(promoted, outputs, command, high(option));
    }
    std::unique_ptr<ncnn::Layer> gpu;
#endif
    std::unique_ptr<ncnn::Layer> cpu;
};
DEFINE_LAYER_CREATOR(ErnieResidualAdd)
} // namespace
int register_residual(ncnn::Net &net)
{
    return net.register_custom_layer("ErnieResidualAdd", ErnieResidualAdd_layer_creator);
}
} // namespace ernie

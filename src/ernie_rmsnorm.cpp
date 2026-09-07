// SPDX-License-Identifier: MIT
#include "ernie_rmsnorm.h"
#include "layer.h"
#include <memory>
#if NCNN_VULKAN
#include "vulkan/rmsnorm_vulkan.h"
#endif

namespace ernie {
namespace {
ncnn::Option fp32(const ncnn::Option& option)
{
    auto result = option;
    result.use_fp16_packed = result.use_fp16_storage = result.use_fp16_arithmetic = false;
    result.use_bf16_packed = result.use_bf16_storage = false;
    return result;
}

// Delegate to the pinned native implementations. In particular, this wrapper
// does not reimplement normalization grouping and packing conventions.
class ErnieNorm : public ncnn::Layer
{
public:
    explicit ErnieNorm(bool centered) : cpu(ncnn::create_layer_cpu(centered ? "LayerNorm" : "RMSNorm")), center(centered)
    {
        one_blob_only = true;
        support_inplace = support_packing = true;
#if NCNN_VULKAN
        support_vulkan = support_vulkan_packing = true;
        gpu.reset(ncnn::create_layer_vulkan(centered ? "LayerNorm" : "RMSNorm"));
#endif
    }

    int load_param(const ncnn::ParamDict& params) override
    {
        affine_size = params.get(0, 0);
        if (!cpu) return -1;
        if (center)
        {
            if (affine_size != 4096 || params.get(2, 1) != 0) return -1;
        }
        else if ((affine_size != 128 && affine_size != 3072 && affine_size != 4096) || params.get(2, 1) != 1) return -1;
        int result = cpu->load_param(params);
#if NCNN_VULKAN
        if (!result) result = gpu ? gpu->load_param(params) : -1;
#endif
        return result;
    }

    int load_model(const ncnn::ModelBin& model) override
    {
        if (center)
        {
            // The ERNIE final LayerNorm is non-affine and consumes no weights.
            int result = cpu->load_model(model);
#if NCNN_VULKAN
            if (!result) result = gpu->load_model(model);
#endif
            return result;
        }
        const ncnn::Mat weight = model.load(affine_size, 1);
        if (weight.empty()) return -100;
        ncnn::ModelBinFromMatArray cpu_model(&weight);
        int result = cpu->load_model(cpu_model);
#if NCNN_VULKAN
        ncnn::ModelBinFromMatArray gpu_model(&weight);
        if (!result) result = gpu->load_model(gpu_model);
#endif
        return result;
    }

    int create_pipeline(const ncnn::Option& option) override
    {
#if NCNN_VULKAN
        if (option.use_vulkan_compute)
        {
            gpu->vkdev = vkdev;
            return gpu->create_pipeline(fp32(option));
        }
#endif
        return cpu->create_pipeline(fp32(option));
    }

    int destroy_pipeline(const ncnn::Option& option) override
    {
        int result = cpu->destroy_pipeline(fp32(option));
#if NCNN_VULKAN
        if (gpu) gpu->destroy_pipeline(fp32(option));
#endif
        return result;
    }

    int forward_inplace(ncnn::Mat& value, const ncnn::Option& option) const override
    {
        if (value.elembits() != 32) return -1;
        return cpu->forward_inplace(value, fp32(option));
    }

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& command, const ncnn::Option& option) override
    {
        return gpu->upload_model(command, fp32(option));
    }

    int forward_inplace(ncnn::VkMat& value, ncnn::VkCompute& command, const ncnn::Option& option) const override
    {
        const auto high = fp32(option);
        ncnn::VkMat promoted, result;
        // Original options identify BF16 versus FP16 source bits. Cast type 1
        // explicitly requests FP32 while preserving the current packing.
        if (value.elembits() == 32) promoted = value;
        else if (value.elembits() == 16)
            vkdev->convert_packing(value, promoted, value.elempack, 1, command, option);
        else return -1;
        if (promoted.empty()) return -100;
        int status = gpu->forward_inplace(promoted, command, high);
        if (status) return status;
        if (!option.use_fp16_storage && !option.use_fp16_packed && !option.use_bf16_storage && !option.use_bf16_packed)
        {
            value = promoted;
            return 0;
        }
        // A FP32 skip path still feeds low-storage projections after norm.
        const int storage_type = option.use_bf16_storage || option.use_bf16_packed ? 5 : 2;
        vkdev->convert_packing(promoted, result, value.elempack, storage_type, command, option);
        if (result.empty()) return -100;
        value = result;
        return 0;
    }
    std::unique_ptr<ncnn::Layer> gpu;
#endif
    std::unique_ptr<ncnn::Layer> cpu;
    const bool center;
    int affine_size = 0;
};
class ErnieRMSNorm final : public ErnieNorm
{
public:
    ErnieRMSNorm() : ErnieNorm(false) {}
};
class ErnieLayerNorm final : public ErnieNorm
{
public:
    ErnieLayerNorm() : ErnieNorm(true) {}
};
DEFINE_LAYER_CREATOR(ErnieRMSNorm)
DEFINE_LAYER_CREATOR(ErnieLayerNorm)
} // namespace

int register_rmsnorm(ncnn::Net& net)
{
    return net.register_custom_layer("RMSNorm", ErnieRMSNorm_layer_creator);
}
int register_layernorm(ncnn::Net& net)
{
    return net.register_custom_layer("LayerNorm", ErnieLayerNorm_layer_creator);
}
#if NCNN_VULKAN
void append_rmsnorm_weights(const ncnn::Layer& layer, std::vector<ncnn::VkMat>& gpu,
                           std::vector<ncnn::Mat>& cpu)
{
    const auto& norm = static_cast<const ErnieRMSNorm&>(layer);
    const auto& native = *static_cast<const ncnn::RMSNorm_vulkan*>(norm.gpu.get());
    gpu.push_back(native.gamma_data_gpu);
    cpu.push_back(native.gamma_data);
    cpu.push_back(static_cast<const ncnn::RMSNorm*>(norm.cpu.get())->gamma_data);
}
#endif
} // namespace ernie

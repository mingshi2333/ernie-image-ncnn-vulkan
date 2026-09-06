// SPDX-License-Identifier: MIT
#include "model_config.h"
#include "gpu_context.h"
#include <ernie/pipeline.h>
#include <platform.h>
#if NCNN_VULKAN
#include "gpu.h"
#endif
#include <filesystem>

namespace ernie
{
DiagnosticInfo diagnose(const std::string &model_directory)
{
    DiagnosticInfo result;
#if NCNN_VULKAN
    result.vulkan_compiled = true;
    GpuContext gpu(true, -1, false);
    result.vulkan_error = gpu.initialization_error();
    if (gpu.available())
    {
        const int count = ncnn::get_gpu_count();
        if (!count)
            result.vulkan_error = "No Vulkan device";
        result.default_gpu_index = count ? ncnn::get_default_gpu_index() : -1;
        for (int index = 0; index < count; ++index)
        {
            const auto &info = ncnn::get_gpu_info(index);
            result.vulkan_devices.push_back(
                {index, info.device_name(), info.support_fp16_storage(), info.support_bf16_storage()});
        }
    }
#endif
    if (!model_directory.empty())
    {
        const auto config = model_config(std::filesystem::path(model_directory) / "model.cfg");
        result.model_config_loaded = true;
        result.packed_width = config.packed_width;
        result.packed_height = config.packed_height;
        result.text_bucket = config.text_bucket;
        result.dit_text_tokens = config.dit_text_tokens;
        result.text_layers = 25;
        result.dit_layers = 36;
    }
    return result;
}
} // namespace ernie

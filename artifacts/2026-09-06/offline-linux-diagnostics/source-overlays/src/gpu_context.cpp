// SPDX-License-Identifier: MIT
#include "gpu_context.h"
#include <platform.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#include <mutex>
#include <stdexcept>

namespace ernie
{
namespace
{
std::mutex context_mutex;
size_t context_users = 0;
bool context_owned = false;
}

GpuContext::GpuContext(bool enabled, int requested_index, bool require_device)
{
    if (!enabled)
        return;
#if NCNN_VULKAN
    const std::lock_guard<std::mutex> lock(context_mutex);
    const bool first = context_users == 0;
    if (first)
    {
        // ncnn exposes a non-creating query. Respect an application's existing
        // instance; its owner must keep it alive throughout the borrowed call.
        context_owned = ncnn::get_gpu_instance() == VK_NULL_HANDLE;
        if (context_owned && ncnn::create_gpu_instance())
        {
            ncnn::destroy_gpu_instance();
            context_owned = false;
            if (require_device)
                throw std::runtime_error("Cannot initialize Vulkan");
            initialization_error_ = "Cannot initialize Vulkan";
            return;
        }
    }
    try
    {
        const int count = ncnn::get_gpu_count();
        if (require_device && count < 1)
            throw std::runtime_error("No Vulkan device");
        if (requested_index < -1 || (requested_index >= 0 && requested_index >= count))
            throw std::invalid_argument("Requested Vulkan GPU index is unavailable");
    }
    catch (...)
    {
        if (first && context_owned)
        {
            ncnn::destroy_gpu_instance();
            context_owned = false;
        }
        throw;
    }
    ++context_users;
    enabled_ = true;
#else
    (void)requested_index;
    (void)require_device;
    throw std::runtime_error("Built without Vulkan");
#endif
}

GpuContext::~GpuContext()
{
#if NCNN_VULKAN
    if (enabled_)
    {
        const std::lock_guard<std::mutex> lock(context_mutex);
        if (--context_users == 0 && context_owned)
        {
            ncnn::destroy_gpu_instance();
            context_owned = false;
        }
    }
#endif
}
} // namespace ernie

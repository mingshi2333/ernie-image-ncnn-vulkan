// SPDX-License-Identifier: MIT
#pragma once

#include "gpu.h"
#include <new>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ncnn {
namespace ernie_ncnn_detail {

// Only original Vulkan result codes enter this function, never ncnn's -1.
inline void throw_allocation_failure(VkResult result, const char* operation)
{
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        throw std::bad_alloc();
    if (result == VK_ERROR_DEVICE_LOST)
        throw std::runtime_error(std::string(operation) + ": Vulkan device lost");
}

inline void check(VkResult result, const char* operation)
{
    throw_allocation_failure(result, operation);
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}

inline std::recursive_mutex& submission_mutex()
{
    static std::recursive_mutex mutex;
    return mutex;
}

// Both command types share this lock, including on single-queue devices.
// CPU preparation can overlap compute; submission and waiting are serialized.
class QueueLease
{
public:
    QueueLease(const VulkanDevice* device, uint32_t family)
        : lock_(submission_mutex()), device_(device), family_(family) {}
    ~QueueLease() { if (queue_) device_->reclaim_queue(family_, queue_); }
    QueueLease(const QueueLease&) = delete;
    QueueLease& operator=(const QueueLease&) = delete;
    VkQueue acquire() { queue_ = device_->acquire_queue(family_); return queue_; }
    void drain() const
    {
        if (!queue_) return;
        const VkResult result = vkQueueWaitIdle(queue_);
        // Failed draining is fatal: buffers cannot be released and retried
        // while their previous submission may still be using them.
        if (result != VK_SUCCESS)
            throw std::runtime_error("Vulkan queue drain failed before memory recovery: " + std::to_string(result));
    }
private:
    std::unique_lock<std::recursive_mutex> lock_;
    const VulkanDevice* device_;
    uint32_t family_;
    VkQueue queue_ = 0;
};

inline void submission_failure(VkResult result, const QueueLease& compute,
                               const QueueLease& transfer)
{
    if (result == VK_ERROR_DEVICE_LOST)
        throw std::runtime_error("Vulkan submission or fence wait: device lost");
    transfer.drain();
    compute.drain();
    throw_allocation_failure(result, "Vulkan submission or fence wait");
}

} // namespace ernie_ncnn_detail
} // namespace ncnn

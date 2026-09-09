// SPDX-License-Identifier: MIT
#pragma once
#include "host_memory.h"
#include "weight_placement.h"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace ernie {
enum class ActivationMemory { Auto, Device, Host };
ActivationMemory parse_activation_memory(const std::string& value);

class GpuAllocationError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
// Keep the two return-code domains separate: ncnn -1 is not Vulkan OOM.
void check_ncnn_memory(int result, const char* action);
void check_vulkan_memory(int result, const char* action);

#if NCNN_VULKAN
// Query the compute heap from a storage buffer's requirements without allocating
// device memory. This remains usable while recovering an exhausted device heap.
WeightPlacement::BudgetReader compute_memory_budget_reader(const ncnn::VulkanDevice* device);

struct VulkanMemoryStats
{
    // Successful allocator requests, not unique underlying device allocations.
    std::uint64_t device_allocations = 0, host_allocations = 0;
    std::uint64_t fallbacks = 0, allocation_failures = 0;
    // Actual VkDeviceMemory allocation sizes; retired buffers remain charged
    // until their GPU command has completed and reclaim_completed() runs.
    std::uint64_t host_live_bytes = 0, host_peak_bytes = 0;
    // Unified-memory devices can have only host-visible device-local types.
    std::uint64_t host_device_local_allocations = 0, host_non_device_local_allocations = 0;
};

// One compute stream owns this allocator. GPU commands and every VkMat must
// finish before it is destroyed. Host buffers remain ordinary Vulkan storage:
// all arithmetic stays on the GPU, with the same shaders and tensor shapes.
class AdaptiveVkAllocator : public ncnn::VkAllocator
{
public:
    using BudgetReader = WeightPlacement::BudgetReader;
    AdaptiveVkAllocator(const ncnn::VulkanDevice* device, ActivationMemory mode,
        std::uint64_t gpu_reserve_bytes, std::uint64_t host_limit_bytes,
        std::uint64_t ram_reserve_bytes,
        HostAvailableReader reader = host_memory_available_reader(), BudgetReader budget_reader = {});
    ~AdaptiveVkAllocator() override;
    AdaptiveVkAllocator(const AdaptiveVkAllocator&) = delete;
    AdaptiveVkAllocator& operator=(const AdaptiveVkAllocator&) = delete;

    VulkanMemoryStats stats() const;
    ncnn::VkBufferMemory* fastMalloc(size_t size) override;
    void fastFree(ncnn::VkBufferMemory* ptr) override;
    int flush(ncnn::VkBufferMemory* ptr) override;
    int invalidate(ncnn::VkBufferMemory* ptr) override;
    // The production runtime disables image storage. Image requests retain
    // ncnn's device allocator and ownership; they do not spill into RAM.
    ncnn::VkImageMemory* fastMalloc(int w, int h, int c, size_t elemsize, int elempack) override;
    void fastFree(ncnn::VkImageMemory* ptr) override;

    // Call only after submitted commands complete and their command object is
    // reset/destroyed (or unsubmitted commands are destroyed). fastFree alone
    // cannot know whether a recorded command still refers to a buffer.
    void reclaim_completed();
    // Same completed-command requirement; active tensors remain allocated.
    void clear() override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// No-op for ncnn allocators, so synchronization sites can keep their existing
// caller-provided allocator support.
void reclaim_completed(ncnn::VkAllocator* allocator);
#endif
} // namespace ernie

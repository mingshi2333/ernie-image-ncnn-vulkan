// SPDX-License-Identifier: MIT
#include "weight_placement.h"
#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ernie {
WeightMemory parse_weight_memory(const std::string& value)
{
    if (value == "auto") return WeightMemory::Auto;
    if (value == "device") return WeightMemory::Device;
    if (value == "host") return WeightMemory::Host;
    throw std::invalid_argument("DiT weights must be auto, device, or host");
}

WeightPlacementDecision choose_weight_placement(WeightMemory mode, bool shape_prefers_host,
    std::uint64_t weight_bytes, std::uint64_t reserve_bytes,
    std::optional<DeviceMemoryBudget> memory)
{
    WeightPlacementDecision d;
    d.weight_bytes = weight_bytes;
    d.reserve_bytes = reserve_bytes;
    if (memory)
        d.available_bytes = memory->usage_bytes < memory->budget_bytes ?
            memory->budget_bytes - memory->usage_bytes : 0;
    if (mode != WeightMemory::Auto)
    {
        d.host = mode == WeightMemory::Host;
        d.reason = "explicit";
    }
    else if (shape_prefers_host)
    {
        d.host = true;
        d.reason = "shape";
    }
    else if (d.available_bytes)
    {
        // Subtraction avoids overflow even for a caller's very large reserve.
        d.host = *d.available_bytes < reserve_bytes ||
            *d.available_bytes - reserve_bytes < weight_bytes;
        d.reason = d.host ? "budget" : "available";
    }
    return d;
}

WeightPlacement::WeightPlacement(WeightMemory mode, std::uint64_t reserve_bytes,
                               BudgetReader reader, Observer observer)
    : mode_(mode), reserve_bytes_(reserve_bytes), reader_(std::move(reader)), observer_(std::move(observer)) {}

bool WeightPlacement::use_host(const ComponentFiles& files, bool shape_prefers_host, bool cache_prefers_host,
                               const char* host_reason)
{
    // Reviewed packages store BF16 or FP32 weights. Twice the on-disk bytes is
    // a conservative payload estimate at FP32, including for lower storage
    // precision. It is not an estimate of activation or allocator peak memory.
    const auto file_bytes = std::filesystem::file_size(std::filesystem::u8path(files.weight_path));
    if (file_bytes > std::numeric_limits<std::uint64_t>::max() / 2)
        throw std::overflow_error("Weight byte estimate overflow");
    const auto memory = mode_ == WeightMemory::Auto && reader_ ? reader_() : std::nullopt;
    auto d = choose_weight_placement(mode_, shape_prefers_host, file_bytes * 2, reserve_bytes_, memory);
    if (mode_ == WeightMemory::Auto && cache_prefers_host)
    {
        d.host = true;
        d.reason = host_reason;
    }
    if (mode_ == WeightMemory::Auto && !memory) ++unavailable_queries_;
    if (d.host) ++host_requests_; else ++device_requests_;
    if (observer_) observer_(files, d);
    return d.host;
}

#if NCNN_VULKAN
WeightPlacement::BudgetReader vulkan_memory_budget_reader(const ncnn::VulkanDevice* device,
                                                         std::uint32_t memory_type_index)
{
    if (!device) throw std::invalid_argument("Memory budget requires a Vulkan device");
    const auto& properties = device->info.physicalDeviceMemoryProperties();
    if (memory_type_index >= properties.memoryTypeCount)
        throw std::invalid_argument("Invalid activation memory type");
    const auto heap = properties.memoryTypes[memory_type_index].heapIndex;
    const auto capacity = properties.memoryHeaps[heap].size;
    const bool supported = device->info.support_VK_EXT_memory_budget() &&
        (properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
        ncnn::vkGetPhysicalDeviceMemoryProperties2KHR;
    return [device, heap, capacity, supported]() -> std::optional<DeviceMemoryBudget> {
        if (!supported) return std::nullopt;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2KHR properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
        properties.pNext = &budget;
        ncnn::vkGetPhysicalDeviceMemoryProperties2KHR(device->info.physicalDevice(), &properties);
        return DeviceMemoryBudget{std::min(capacity, budget.heapBudget[heap]), budget.heapUsage[heap]};
    };
}
#endif
} // namespace ernie

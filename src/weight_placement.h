// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "platform.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif
#include <cstdint>
#include <functional>
#include <optional>

namespace ernie {
enum class WeightMemory { Auto, Device, Host };
WeightMemory parse_weight_memory(const std::string& value);

struct DeviceMemoryBudget
{
    std::uint64_t budget_bytes = 0, usage_bytes = 0;
};
struct WeightPlacementDecision
{
    bool host = false;
    const char* reason = "unavailable";
    std::optional<std::uint64_t> available_bytes;
    std::uint64_t weight_bytes = 0, reserve_bytes = 0;
};
WeightPlacementDecision choose_weight_placement(WeightMemory mode, bool shape_prefers_host,
    std::uint64_t weight_bytes, std::uint64_t reserve_bytes,
    std::optional<DeviceMemoryBudget> memory);

// Decisions apply before loading a new Net, after preceding GPU work completes.
// This owns neither weights nor GPU commands. Counts describe requests: ncnn's
// allocator can itself fall back from host to device if host allocation fails.
class WeightPlacement
{
public:
    using BudgetReader = std::function<std::optional<DeviceMemoryBudget>()>;
    using Observer = std::function<void(const ComponentFiles&, const WeightPlacementDecision&)>;
    WeightPlacement(WeightMemory mode, std::uint64_t reserve_bytes,
                    BudgetReader reader = {}, Observer observer = {});
    bool use_host(const ComponentFiles& files, bool shape_prefers_host, bool cache_prefers_host = false,
                  const char* host_reason = "cache");
    std::uint64_t host_requests() const { return host_requests_; }
    std::uint64_t device_requests() const { return device_requests_; }
    std::uint64_t unavailable_queries() const { return unavailable_queries_; }
private:
    WeightMemory mode_;
    std::uint64_t reserve_bytes_, host_requests_ = 0, device_requests_ = 0, unavailable_queries_ = 0;
    BudgetReader reader_;
    Observer observer_;
};

#if NCNN_VULKAN
// The memory type comes from an actual session activation on this device, so
// the query addresses the compute heap rather than a guessed largest heap.
WeightPlacement::BudgetReader vulkan_memory_budget_reader(const ncnn::VulkanDevice* device,
                                                         std::uint32_t memory_type_index);
#endif
} // namespace ernie

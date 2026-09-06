// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include "execution_metrics.h"
namespace ernie {
struct VulkanMemoryClass {
    std::uint32_t type_index=0, heap_index=0, property_flags=0;
    bool imported_host=false;
    std::uint32_t heap_flags=0;
    bool operator<(const VulkanMemoryClass& b) const {
        return std::tie(type_index,heap_index,property_flags,imported_host,heap_flags)<
               std::tie(b.type_index,b.heap_index,b.property_flags,b.imported_host,b.heap_flags);
    }
};
struct AllocationHookSnapshot {
    bool available=false, valid=true;
    // Complete only for observed ncnn allocator events during this session.
    // This is never a claim of total driver/process/device memory coverage.
    std::optional<AllocationTotals> all_memory;
    std::map<std::pair<std::uint64_t,VulkanMemoryClass>,AllocationTotals> by_device_memory_class;
    MetricsSnapshot allocator_metrics;
};
class AllocationMeasurementSession {
public:
    AllocationMeasurementSession();
    ~AllocationMeasurementSession();
    AllocationMeasurementSession(const AllocationMeasurementSession&)=delete;
    AllocationMeasurementSession& operator=(const AllocationMeasurementSession&)=delete;
    AllocationHookSnapshot snapshot() const;
private:
    struct State;
    std::unique_ptr<State> state_;
    friend struct AllocationHookAccess;
};
// These observers never return a changed Vulkan result and never throw across ncnn.
// The patched source must still perform its original allocation/free unconditionally.
void allocation_allocator_created(std::uint64_t device,std::uint64_t allocator) noexcept;
void allocation_allocator_role(std::uint64_t device,std::uint64_t allocator,AllocationRole) noexcept;
void allocation_allocator_destroyed(std::uint64_t device,std::uint64_t allocator) noexcept;
void allocation_memory_created(std::uint64_t device,std::uint64_t allocator,std::uint64_t handle,
                               std::uint64_t allocation_size,VulkanMemoryClass) noexcept;
void allocation_memory_destroyed(std::uint64_t device,std::uint64_t allocator,std::uint64_t handle) noexcept;
}

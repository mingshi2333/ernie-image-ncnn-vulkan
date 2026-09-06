// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <array>
#include <string>
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
struct AllocationDeviceIdentity {
    int index=-1;
    std::uint32_t vendor_id=0, device_id=0, api_version=0, driver_version=0;
    std::string name;
    std::array<std::uint8_t,16> pipeline_cache_uuid{};
    bool operator==(const AllocationDeviceIdentity& b) const {
        return std::tie(index,vendor_id,device_id,api_version,driver_version,name,pipeline_cache_uuid)==
               std::tie(b.index,b.vendor_id,b.device_id,b.api_version,b.driver_version,b.name,b.pipeline_cache_uuid);
    }
};
struct AllocationHookSnapshot {
    bool available=false, valid=true;
    // Complete only for observed ncnn allocator events during this session.
    // This is never a claim of total driver/process/device memory coverage.
    std::optional<AllocationTotals> all_memory;
    std::map<std::pair<std::uint64_t,VulkanMemoryClass>,AllocationTotals> by_device_memory_class;
    MetricsSnapshot allocator_metrics;
    std::map<std::uint64_t,AllocationDeviceIdentity> devices;
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
// Copies identity from an already existing GpuInfo; never creates a device.
void allocation_device_identified(std::uint64_t device,int index,std::uint32_t vendor_id,
    std::uint32_t device_id,std::uint32_t api_version,std::uint32_t driver_version,
    const char* name,const std::uint8_t* pipeline_cache_uuid) noexcept;
void allocation_allocator_created(std::uint64_t device,std::uint64_t allocator) noexcept;
void allocation_allocator_role(std::uint64_t device,std::uint64_t allocator,AllocationRole) noexcept;
void allocation_allocator_destroyed(std::uint64_t device,std::uint64_t allocator) noexcept;
void allocation_memory_created(std::uint64_t device,std::uint64_t allocator,std::uint64_t handle,
                               std::uint64_t allocation_size,VulkanMemoryClass) noexcept;
void allocation_memory_destroyed(std::uint64_t device,std::uint64_t allocator,std::uint64_t handle) noexcept;
}

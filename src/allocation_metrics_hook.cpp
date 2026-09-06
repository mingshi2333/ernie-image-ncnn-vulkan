// SPDX-License-Identifier: MIT
#include "allocation_metrics_hook.h"
#include <limits>
#include <atomic>
#include <type_traits>
#include <mutex>
#include <stdexcept>
#ifndef ERNIE_ALLOCATION_HOOKS_ENABLED
#define ERNIE_ALLOCATION_HOOKS_ENABLED 0
#endif
namespace ernie {
namespace {
std::mutex observer_mutex;
void* active_state=nullptr;
std::atomic<bool> observer_failed{false};
using Key=std::pair<std::uint64_t,std::uint64_t>;
void increment(AllocationTotals& t,std::uint64_t bytes) {
    if (bytes>std::numeric_limits<std::uint64_t>::max()-t.live_bytes || t.allocations==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("allocation metrics");
    t.live_bytes+=bytes;++t.allocations;if(t.peak_bytes<t.live_bytes)t.peak_bytes=t.live_bytes;
}
}
struct AllocationMeasurementSession::State {
    struct Allocator {AllocatorIdentity identity;AllocationRole role=AllocationRole::Other;};
    struct Memory {Key owner;std::uint64_t bytes;VulkanMemoryClass type;};
    bool valid=true, observed=false;
    ExecutionMetrics metrics;
    std::map<Key,Allocator> allocators;
    // Device+VkDeviceMemory is unique even when allocator pointers are reused.
    std::map<Key,Memory> memories;
    AllocationTotals total;
    std::map<std::pair<std::uint64_t,VulkanMemoryClass>,AllocationTotals> classes;
};
struct AllocationHookAccess {
    template<class F> static void observe(F fn) noexcept {
        try {
            std::lock_guard<std::mutex> lock(observer_mutex);
            auto* s=static_cast<AllocationMeasurementSession::State*>(active_state);
            if (!ERNIE_ALLOCATION_HOOKS_ENABLED || !s || !s->valid) return;
            try {fn(*s);} catch (...) {s->valid=false;}
        } catch (...) { observer_failed.store(true); }
    }
};
AllocationMeasurementSession::AllocationMeasurementSession():state_(new State) {
    std::lock_guard<std::mutex> lock(observer_mutex);
    if (active_state) throw std::logic_error("allocation measurement session already active");
    active_state=state_.get();observer_failed.store(false);
}
AllocationMeasurementSession::~AllocationMeasurementSession() {
    std::lock_guard<std::mutex> lock(observer_mutex);active_state=nullptr;
}
AllocationHookSnapshot AllocationMeasurementSession::snapshot() const {
    std::lock_guard<std::mutex> lock(observer_mutex);AllocationHookSnapshot r;
    r.available=ERNIE_ALLOCATION_HOOKS_ENABLED;r.valid=state_->valid && !observer_failed.load();
    if (r.available) {
        if(state_->observed)r.all_memory=state_->total;
        r.by_device_memory_class=state_->classes;
        r.allocator_metrics=state_->metrics.snapshot();
    }
    return r;
}
void allocation_allocator_created(std::uint64_t device,std::uint64_t allocator) noexcept {
    AllocationHookAccess::observe([&](auto& s){
        Key key{device,allocator};if(s.allocators.count(key))throw std::logic_error("duplicate allocator");
        auto id=s.metrics.register_allocator(device,allocator,AllocationRole::Other,AllocationDomain::VulkanMemory,"observed ncnn VkDeviceMemory allocations; excludes other driver allocations");
        s.observed=true;
        s.allocators.emplace(key,typename std::decay_t<decltype(s)>::Allocator{id});
    });
}
void allocation_allocator_role(std::uint64_t device,std::uint64_t allocator,AllocationRole role) noexcept {
    AllocationHookAccess::observe([&](auto& s){
        auto& a=s.allocators.at({device,allocator});
        // Role tagging occurs in derived constructors before first allocation.
        s.metrics.unregister_allocator(a.identity);
        a.identity=s.metrics.register_allocator(device,allocator,role,AllocationDomain::VulkanMemory,"observed ncnn VkDeviceMemory allocations; excludes other driver allocations");a.role=role;
    });
}
void allocation_allocator_destroyed(std::uint64_t device,std::uint64_t allocator) noexcept {
    AllocationHookAccess::observe([&](auto& s){auto it=s.allocators.find({device,allocator});if(it==s.allocators.end())throw std::logic_error("unknown allocator");s.metrics.unregister_allocator(it->second.identity);s.allocators.erase(it);});
}
void allocation_memory_created(std::uint64_t device,std::uint64_t allocator,std::uint64_t handle,std::uint64_t size,VulkanMemoryClass type) noexcept {
    AllocationHookAccess::observe([&](auto& s){
        auto& a=s.allocators.at({device,allocator});Key key{device,handle};
        if(s.memories.count(key))throw std::logic_error("duplicate memory");
        s.metrics.allocate(a.identity,handle,size);
        increment(s.total,size);increment(s.classes[{device,type}],size);
        s.memories.emplace(key,typename std::decay_t<decltype(s)>::Memory{{device,allocator},size,type});
    });
}
void allocation_memory_destroyed(std::uint64_t device,std::uint64_t allocator,std::uint64_t handle) noexcept {
    if(!handle)return; // vkFreeMemory(VK_NULL_HANDLE) is a valid no-op.
    AllocationHookAccess::observe([&](auto& s){
        auto it=s.memories.find({device,handle});if(it==s.memories.end() || it->second.owner!=Key{device,allocator})throw std::logic_error("unmatched memory free");
        auto& a=s.allocators.at({device,allocator});s.metrics.release(a.identity,handle);
        s.total.live_bytes-=it->second.bytes;s.classes.at({device,it->second.type}).live_bytes-=it->second.bytes;s.memories.erase(it);
    });
}
}

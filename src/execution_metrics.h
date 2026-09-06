// SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ernie {
enum class AllocationDomain { LogicalRequest, AllocatorBlock, VulkanMemory };
enum class AllocationRole { Weight, Blob, Staging, Cache, Other };
enum class ExecutionPhase { Verify, Read, Prepare, ReadPrepare, Upload, Compute, Wait, Download };
struct AllocationTotals { std::uint64_t live_bytes=0, peak_bytes=0, allocations=0; };
struct PhaseTotals {
    // Sum of recorded intervals, NOT elapsed wall time: overlapping calls may overlap.
    std::uint64_t host_nanoseconds=0, samples=0, gpu_samples=0;
    // Present only when every recorded sample has a genuine GPU duration.
    std::optional<std::uint64_t> gpu_nanoseconds;
};
struct AllocatorIdentity {
    std::uint64_t device=0, allocator=0, generation=0;
    bool operator<(const AllocatorIdentity& b) const {
        return std::tie(device,allocator,generation)<std::tie(b.device,b.allocator,b.generation);
    }
};
struct AllocationRecord { std::uint64_t bytes; std::optional<std::uint64_t> alignment; };
struct AllocatorSnapshot {
    AllocationRole role; AllocationDomain domain; std::string scope;
    AllocationTotals totals; bool active=true;
    std::map<std::uint64_t,AllocationRecord> live;
};
struct ComponentInterval {
    std::string component, boundary, status;
    int absolute_step=-1, block=-1;
    std::uint64_t host_nanoseconds=0;
};
struct MetricsSnapshot {
    // nullopt = unobserved/unavailable; a registered domain with no allocations = zero.
    // Registration does not claim complete process/device coverage; scope is required.
    std::array<std::optional<AllocationTotals>,3> allocations;
    std::array<std::optional<PhaseTotals>,8> phases;
    std::optional<std::uint64_t> submissions, upload_bytes, download_bytes;
    std::map<AllocatorIdentity,AllocatorSnapshot> allocators;
    // Diagnostic child intervals. They describe parts of top-level phases and
    // must never be added to the top-level phase totals.
    std::vector<ComponentInterval> component_intervals;
};
class ExecutionMetrics {
public:
    AllocatorIdentity register_allocator(std::uint64_t device, std::uint64_t allocator,
                                        AllocationRole, AllocationDomain, std::string scope);
    void unregister_allocator(AllocatorIdentity);
    // bytes belongs ONLY to the registered domain; never infer Vulkan allocation
    // size from a Mat request. Reused pool blocks remain live until physically freed.
    void allocate(AllocatorIdentity, std::uint64_t handle, std::uint64_t bytes,
                  std::optional<std::uint64_t> alignment={});
    void release(AllocatorIdentity, std::uint64_t handle);
    // IDs must be globally unique within this collector; explicit timestamps permit
    // deterministic testing and avoid silently choosing an incorrect clock scope.
    void record_interval(std::uint64_t event, ExecutionPhase, std::uint64_t start_ns,
                         std::uint64_t finish_ns, std::optional<std::uint64_t> gpu_ns={});
    void record_io(std::uint64_t event, std::uint64_t submissions,
                   std::uint64_t upload_bytes, std::uint64_t download_bytes);
    // Submission count can be observed when transfer byte counts cannot.
    void record_submissions(std::uint64_t event, std::uint64_t submissions);
    void record_component(std::string component, int absolute_step, int block,
                          std::string boundary, std::uint64_t host_nanoseconds,
                          std::string status);
    MetricsSnapshot snapshot() const;
private:
    struct Allocator { AllocationRole role; AllocationDomain domain; std::string scope;
                       std::map<std::uint64_t,AllocationRecord> live; };
    Allocator& lookup(AllocatorIdentity);
    mutable std::mutex mutex_;
    std::uint64_t generation_=0;
    std::map<AllocatorIdentity,Allocator> allocators_;
    std::map<AllocatorIdentity,AllocatorSnapshot> history_;
    std::set<std::uint64_t> events_;
    MetricsSnapshot totals_;
};
}

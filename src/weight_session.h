// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "host_memory.h"
#include "net.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace ernie {
struct WeightBudget
{
    std::uint64_t host_bytes = 0;
    std::uint64_t reserve_bytes = 3ull * 1024 * 1024 * 1024;
};
struct WeightSessionStats
{
    std::uint64_t hits = 0, loads = 0, admissions = 0, evictions = 0;
    std::uint64_t live_bytes = 0, peak_bytes = 0, cached_nets = 0, peak_nets = 0;
    std::uint64_t unavailable_queries = 0;
};

// A request-local, synchronous prepared-Net cache. Admitted entries stay in
// place while the rest stream: ordinary LRU would miss every block in a repeated
// 36-block scan. Only independently inspected host weights may be retained.
// The byte budget covers charged cached weights, not total process memory or
// the one streamed Net, activation pools, loading temporaries, and driver state.
class WeightSession
{
    struct State;
public:
    using AvailableReader = HostAvailableReader;
    using Inspector = std::function<std::optional<std::uint64_t>(const ncnn::Net&)>;
    using Loader = std::function<std::unique_ptr<ncnn::Net>(bool prefer_host)>;
    class Lease
    {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) noexcept;
        Lease& operator=(Lease&&) noexcept;
        ~Lease();
        ncnn::Net& net() const;
        bool cached() const { return charge_ != 0; }
        // Call only after all commands and extractors using this Net completed.
        // Abandoning a lease discards that Net; it never enters the idle cache.
        void complete();
    private:
        friend class WeightSession;
        Lease(std::shared_ptr<State>, std::size_t, std::unique_ptr<ncnn::Net>, std::uint64_t);
        void release(bool completed);
        std::shared_ptr<State> state_;
        std::size_t block_ = 0;
        std::unique_ptr<ncnn::Net> net_;
        std::uint64_t charge_ = 0;
    };
    WeightSession(WeightBudget, AvailableReader, Inspector);
    ~WeightSession();
    WeightSession(const WeightSession&) = delete;
    WeightSession& operator=(const WeightSession&) = delete;
    // Identity binds graphs/weights and the immutable request's Option object.
    // One lease at a time. The loader may consume an independently prepared
    // Net; only this owning thread can mutate cache admission or eviction.
    Lease acquire(std::size_t block, const ComponentFiles&, const void* request_identity,
                  std::uint64_t estimated_bytes, const Loader&);
    void cancel(); // Drop idle entries; an active lease retains its Net until released.
    WeightSessionStats stats() const;
    bool contains(std::size_t block) const;
private:
    std::shared_ptr<State> state_;
};

#if NCNN_VULKAN
// Fixed ncnn FP32 DiT graphs only. Reject unknown/low-storage graphs and any
// weight in DEVICE_LOCAL memory, including a host allocator's GPU fallback.
WeightSession::Inspector dit_host_weight_inspector(const ncnn::VulkanDevice*);
#endif
} // namespace ernie

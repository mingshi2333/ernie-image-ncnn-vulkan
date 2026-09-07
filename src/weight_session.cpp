// SPDX-License-Identifier: MIT
#include "weight_session.h"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#if NCNN_VULKAN
#include "ernie_rmsnorm.h"
#include "vulkan/gemm_vulkan.h"
#include <set>
#endif

namespace ernie {
namespace {
std::uint64_t add(std::uint64_t a, std::uint64_t b)
{
    if (b > UINT64_MAX - a) throw std::overflow_error("Weight cache byte count overflow");
    return a + b;
}
} // namespace
struct WeightSession::State
{
    struct Entry { std::unique_ptr<ncnn::Net> net; std::uint64_t charge; };
    WeightBudget budget;
    AvailableReader available;
    Inspector inspect;
    WeightSessionStats stats;
    std::map<std::size_t, Entry> idle;
    std::map<std::size_t, ComponentFiles> identities;
    const void* request = nullptr;
    bool active = false, cancelled = false;

    bool headroom(std::uint64_t extra)
    {
        const auto bytes = available ? available() : std::nullopt;
        if (!bytes) { ++stats.unavailable_queries; return false; }
        return *bytes >= budget.reserve_bytes && *bytes - budget.reserve_bytes >= extra;
    }
    void evict_one()
    {
        auto last = std::prev(idle.end());
        stats.live_bytes -= last->second.charge;
        --stats.cached_nets;
        idle.erase(last);
        ++stats.evictions;
    }
};

WeightSession::WeightSession(WeightBudget budget, AvailableReader available, Inspector inspect)
    : state_(std::make_shared<State>())
{
    state_->budget = budget;
    state_->available = std::move(available);
    state_->inspect = std::move(inspect);
}
WeightSession::~WeightSession() { cancel(); }
WeightSessionStats WeightSession::stats() const { return state_->stats; }
void WeightSession::cancel()
{
    state_->cancelled = true;
    while (!state_->idle.empty()) state_->evict_one();
}

WeightSession::Lease WeightSession::acquire(std::size_t block, const ComponentFiles& files,
    const void* request, std::uint64_t estimate, const Loader& loader)
{
    auto& s = *state_;
    if (s.cancelled || s.active) throw std::logic_error("Weight session cancelled or lease still active");
    if (block >= 36 || !request || !loader || files.empty())
        throw std::invalid_argument("Invalid weight session request");
    if (s.request && s.request != request) throw std::logic_error("Weight session request identity changed");
    s.request = request;
    const auto identity = s.identities.find(block);
    if (identity != s.identities.end() &&
        (identity->second.param_text != files.param_text || identity->second.weight_path != files.weight_path))
        throw std::logic_error("Weight session model identity changed");
    s.identities.emplace(block, files);
    // Recheck between completed components; never evict a Net in flight.
    if (s.budget.host_bytes)
        while (!s.idle.empty() && !s.headroom(s.idle.count(block) ? 0 : estimate)) s.evict_one();
    auto found = s.idle.find(block);
    if (found != s.idle.end())
    {
        auto entry = std::move(found->second);
        s.idle.erase(found);
        ++s.stats.hits;
        s.active = true;
        return Lease(state_, block, std::move(entry.net), entry.charge);
    }
    const bool candidate = s.inspect && estimate <= s.budget.host_bytes - s.stats.live_bytes &&
        s.budget.host_bytes && s.headroom(estimate);
    auto net = loader(candidate);
    if (!net) throw std::runtime_error("Weight loader returned no Net");
    ++s.stats.loads;
    std::uint64_t charge = 0;
    if (candidate)
    {
        const auto measured = s.inspect(*net);
        if (measured && *measured && *measured <= s.budget.host_bytes - s.stats.live_bytes && s.headroom(0))
        {
            charge = *measured;
            s.stats.live_bytes += charge;
            s.stats.peak_bytes = std::max(s.stats.peak_bytes, s.stats.live_bytes);
            ++s.stats.cached_nets;
            s.stats.peak_nets = std::max(s.stats.peak_nets, s.stats.cached_nets);
            ++s.stats.admissions;
        }
    }
    s.active = true;
    return Lease(state_, block, std::move(net), charge);
}

WeightSession::Lease::Lease(std::shared_ptr<State> state, std::size_t block,
    std::unique_ptr<ncnn::Net> net, std::uint64_t charge)
    : state_(std::move(state)), block_(block), net_(std::move(net)), charge_(charge) {}
WeightSession::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)), block_(other.block_), net_(std::move(other.net_)), charge_(other.charge_) {}
WeightSession::Lease& WeightSession::Lease::operator=(Lease&& other) noexcept
{
    if (this != &other)
    {
        release(false);
        state_ = std::move(other.state_);
        block_ = other.block_; net_ = std::move(other.net_); charge_ = other.charge_;
    }
    return *this;
}
WeightSession::Lease::~Lease() { release(false); }
ncnn::Net& WeightSession::Lease::net() const
{
    if (!net_) throw std::logic_error("Weight lease already released");
    return *net_;
}
void WeightSession::Lease::complete()
{
    if (!net_) throw std::logic_error("Weight lease already released");
    release(true);
}
void WeightSession::Lease::release(bool completed)
{
    if (!net_) return;
    auto& s = *state_;
    if (completed && charge_ && !s.cancelled)
    {
        try { s.idle.emplace(block_, State::Entry{std::move(net_), charge_}); }
        catch (...) { s.stats.live_bytes -= charge_; --s.stats.cached_nets; s.active = false; throw; }
    }
    else
    {
        net_.reset();
        s.stats.live_bytes -= charge_;
        if (charge_) --s.stats.cached_nets;
    }
    s.active = false;
    state_.reset();
}

#if NCNN_VULKAN
WeightSession::Inspector dit_host_weight_inspector(const ncnn::VulkanDevice* device)
{
    if (!device) throw std::invalid_argument("Host weight inspection requires Vulkan device");
    return [device](const ncnn::Net& net) -> std::optional<std::uint64_t> {
        const auto& opt = net.opt;
        if (!opt.use_vulkan_compute || !opt.use_weights_in_host_memory || !opt.lightmode ||
            opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
            return std::nullopt;
        std::vector<ncnn::VkMat> weights;
        std::vector<ncnn::Mat> cpu_weights;
        const std::set<std::string> weightless{"Input", "Split", "BinaryOp", "Reshape", "Permute",
            "Slice", "Concat", "SDPA", "ErnieResidualAdd", "ErnieGELU"};
        for (const auto* layer : net.layers())
        {
            if (layer->type == "Gemm" && layer->support_vulkan)
            {
                const auto& gemm = *static_cast<const ncnn::Gemm_vulkan*>(layer);
                if (gemm.int8_scale_term) return std::nullopt;
                weights.insert(weights.end(), {gemm.A_data_gpu, gemm.B_data_gpu, gemm.C_data_gpu});
                cpu_weights.insert(cpu_weights.end(), {gemm.A_data, gemm.B_data, gemm.C_data,
                    gemm.A_data_packed, gemm.B_data_packed, gemm.C_data_packed});
            }
            else if (layer->type == "RMSNorm") append_rmsnorm_weights(*layer, weights, cpu_weights);
            else if (!weightless.count(layer->type)) return std::nullopt;
        }
        // Pinned VkWeightAllocator gives each backing buffer its own memory
        // allocation sized by this Vulkan query. Count shared suballocations once.
        std::set<VkDeviceMemory> allocations;
        std::set<const void*> cpu_allocations;
        std::uint64_t bytes = 0;
        for (const auto& weight : weights)
        {
            if (weight.empty()) continue;
            if (device->is_device_local(weight.data->memory_type_index)) return std::nullopt;
            if (allocations.insert(weight.data->memory).second)
            {
                VkMemoryRequirements requirements;
                ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), weight.data->buffer, &requirements);
                bytes = add(bytes, requirements.size);
            }
        }
        if (allocations.empty()) return std::nullopt;
        for (const auto& weight : cpu_weights)
            if (!weight.empty() && cpu_allocations.insert(weight.data).second)
                bytes = add(bytes, weight.total() * weight.elemsize);
        // Explicit accounting margin for Net/graph/pipeline/host allocator
        // overhead. This is not a measured process-RSS or driver-allocation cap.
        return add(bytes, 64ull * 1024 * 1024);
    };
}
#endif
} // namespace ernie

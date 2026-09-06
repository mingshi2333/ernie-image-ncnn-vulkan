// SPDX-License-Identifier: MIT
#include "execution_metrics.h"
#include <limits>
#include <stdexcept>
#include <utility>
namespace ernie {
namespace {
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if (b>std::numeric_limits<std::uint64_t>::max()-a) throw std::overflow_error("metrics overflow");
    return a+b;
}
template<class E> std::size_t index(E e,std::size_t count) {
    auto i=static_cast<std::size_t>(e);
    if (i>=count) throw std::invalid_argument("invalid metrics enum");
    return i;
}
}
ExecutionMetrics::Allocator& ExecutionMetrics::lookup(AllocatorIdentity id) {
    auto it=allocators_.find(id);
    if (it==allocators_.end()) throw std::invalid_argument("unknown or stale allocator identity");
    return it->second;
}
AllocatorIdentity ExecutionMetrics::register_allocator(std::uint64_t device,std::uint64_t allocator,
    AllocationRole role,AllocationDomain domain,std::string scope) {
    auto d=index(domain,3);index(role,5);
    if (!device || !allocator || scope.empty()) throw std::invalid_argument("missing allocator identity/scope");
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& a:allocators_)
        if (a.first.device==device && a.first.allocator==allocator)
            throw std::invalid_argument("allocator already registered");
    auto generation=add(generation_,1);AllocatorIdentity id{device,allocator,generation};
    allocators_.emplace(id,Allocator{role,domain,scope,{}});
    history_.emplace(id,AllocatorSnapshot{role,domain,std::move(scope),{},true,{}});generation_=generation;
    if (!totals_.allocations[d]) totals_.allocations[d]=AllocationTotals{};
    return id;
}
void ExecutionMetrics::unregister_allocator(AllocatorIdentity id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!lookup(id).live.empty()) throw std::logic_error("allocator still owns live allocations");
    allocators_.erase(id);history_.at(id).active=false;
}
void ExecutionMetrics::allocate(AllocatorIdentity id,std::uint64_t handle,std::uint64_t bytes,std::optional<std::uint64_t> alignment) {
    if (!handle || !bytes) throw std::invalid_argument("zero allocation handle/size");
    if (alignment && (!*alignment || (*alignment & (*alignment-1)))) throw std::invalid_argument("invalid allocation alignment");
    std::lock_guard<std::mutex> lock(mutex_);auto& allocator=lookup(id);
    if (allocator.live.count(handle)) throw std::invalid_argument("duplicate live allocation");
    auto& current=*totals_.allocations[index(allocator.domain,3)];auto next=current;
    next.live_bytes=add(next.live_bytes,bytes);next.allocations=add(next.allocations,1);
    if (next.live_bytes>next.peak_bytes) next.peak_bytes=next.live_bytes;
    auto local=history_.at(id).totals;
    local.live_bytes=add(local.live_bytes,bytes);local.allocations=add(local.allocations,1);
    if (local.live_bytes>local.peak_bytes) local.peak_bytes=local.live_bytes;
    allocator.live.emplace(handle,AllocationRecord{bytes,alignment});current=next;history_.at(id).totals=local;
}
void ExecutionMetrics::release(AllocatorIdentity id,std::uint64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);auto& allocator=lookup(id);
    auto it=allocator.live.find(handle);
    if (it==allocator.live.end()) throw std::invalid_argument("unknown or already freed allocation");
    totals_.allocations[index(allocator.domain,3)]->live_bytes-=it->second.bytes;
    history_.at(id).totals.live_bytes-=it->second.bytes;
    allocator.live.erase(it);
}
void ExecutionMetrics::record_interval(std::uint64_t event,ExecutionPhase phase,
    std::uint64_t start,std::uint64_t finish,std::optional<std::uint64_t> gpu) {
    auto p=index(phase,8);
    if (!event || finish<start) throw std::invalid_argument("invalid interval identity/clocks");
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.count(event)) throw std::invalid_argument("duplicate metrics event");
    auto next=totals_.phases[p].value_or(PhaseTotals{});
    next.host_nanoseconds=add(next.host_nanoseconds,finish-start);
    if (gpu) {
        if (next.samples==next.gpu_samples) next.gpu_nanoseconds=add(next.gpu_nanoseconds.value_or(0),*gpu);
        next.gpu_samples=add(next.gpu_samples,1);
    } else next.gpu_nanoseconds.reset();
    next.samples=add(next.samples,1);
    events_.insert(event);totals_.phases[p]=next;
}
void ExecutionMetrics::record_io(std::uint64_t event,std::uint64_t submissions,std::uint64_t upload,std::uint64_t download) {
    if (!event) throw std::invalid_argument("missing IO event identity");
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.count(event)) throw std::invalid_argument("duplicate metrics event");
    auto s=add(totals_.submissions.value_or(0),submissions);
    auto u=add(totals_.upload_bytes.value_or(0),upload);
    auto d=add(totals_.download_bytes.value_or(0),download);
    events_.insert(event);totals_.submissions=s;totals_.upload_bytes=u;totals_.download_bytes=d;
}
void ExecutionMetrics::record_submissions(std::uint64_t event,std::uint64_t submissions) {
    if(!event) throw std::invalid_argument("missing submission event identity");
    std::lock_guard<std::mutex> lock(mutex_);
    if(events_.count(event)) throw std::invalid_argument("duplicate metrics event");
    totals_.submissions=add(totals_.submissions.value_or(0),submissions);
    events_.insert(event);
}
void ExecutionMetrics::record_component(std::string component,int step,int block,
    std::string boundary,std::uint64_t ns,std::string status) {
    if(component.empty() || boundary.empty() || (status!="complete" && status!="failed"))
        throw std::invalid_argument("invalid component interval");
    if(step < -1 || block < -1) throw std::invalid_argument("invalid component coordinates");
    std::lock_guard<std::mutex> lock(mutex_);
    totals_.component_intervals.push_back({std::move(component),std::move(boundary),std::move(status),step,block,ns});
}
MetricsSnapshot ExecutionMetrics::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);auto result=totals_;result.allocators=history_;
    for (const auto& a:allocators_) result.allocators.at(a.first).live=a.second.live;
    return result;
}
}

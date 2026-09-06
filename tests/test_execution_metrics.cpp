// SPDX-License-Identifier: MIT
#include "execution_metrics.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
using namespace ernie;
void check(bool b) { if (!b) throw std::runtime_error("metrics contract failed"); }
template<class F> void fails(F fn) { bool caught=false;try { fn(); } catch(const std::exception&) { caught=true; }check(caught); }
int main() {
 try {
    ExecutionMetrics m;auto initial=m.snapshot();check(!initial.allocations[2] && !initial.phases[5] && !initial.submissions);
    auto a=m.register_allocator(1,5,AllocationRole::Blob,AllocationDomain::VulkanMemory,"synthetic device-memory allocation events");
    check(m.snapshot().allocations[2]->live_bytes==0);
    m.allocate(a,10,10);m.allocate(a,20,20);m.release(a,10);m.allocate(a,30,15);
    auto s=m.snapshot();check(s.allocations[2]->live_bytes==35 && s.allocations[2]->peak_bytes==35);
    fails([&]{m.release(a,10);});fails([&]{m.release(a,99);});fails([&]{m.allocate(a,20,3);});fails([&]{m.unregister_allocator(a);});
    fails([&]{m.register_allocator(1,5,AllocationRole::Blob,AllocationDomain::VulkanMemory,"duplicate");});
    // Same handle on another device is independent; wrong generation never aliases.
    auto b=m.register_allocator(2,5,AllocationRole::Cache,AllocationDomain::VulkanMemory,"device2 cache");
    m.allocate(b,20,7);check(m.snapshot().allocations[2]->live_bytes==42);m.release(b,20);m.unregister_allocator(b);
    m.release(a,20);m.release(a,30);m.unregister_allocator(a);
    auto reused=m.register_allocator(1,5,AllocationRole::Weight,AllocationDomain::VulkanMemory,"recreated allocator");
    check(reused.generation!=a.generation);fails([&]{m.allocate(a,10,1);});
    m.allocate(reused,10,3);m.release(reused,10);check(!m.snapshot().allocators.at(a).active);
    // Real host allocation + reusable pool leases: request release does NOT free pool backing.
    ExecutionMetrics pool;auto backing=pool.register_allocator(1,1,AllocationRole::Blob,AllocationDomain::AllocatorBlock,"test-owned host pool backing; not Vulkan");
    auto requests=pool.register_allocator(1,2,AllocationRole::Blob,AllocationDomain::LogicalRequest,"test host pool leases");
    auto memory=std::make_unique<unsigned char[]>(64);auto handle=static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(memory.get()));
    pool.allocate(backing,handle,64,16);
    check(pool.snapshot().allocators.at(backing).live.at(handle).alignment==16);
    fails([&]{pool.allocate(backing,handle+1,7,3);});
    for (int i=0;i<2;++i) { pool.allocate(requests,handle,10);memory[0]=static_cast<unsigned char>(i);pool.release(requests,handle); }
    auto p=pool.snapshot();check(p.allocations[0]->live_bytes==0 && p.allocations[0]->peak_bytes==10);
    check(p.allocations[1]->live_bytes==64 && p.allocations[1]->allocations==1 && !p.allocations[2]);
    memory.reset();pool.release(backing,handle);check(pool.snapshot().allocations[1]->live_bytes==0);
    // Missing GPU samples cannot silently become zero or a partial sum presented as complete.
    m.record_interval(1,ExecutionPhase::Compute,100,130);s=m.snapshot();check(s.phases[5]->host_nanoseconds==30 && !s.phases[5]->gpu_nanoseconds);
    m.record_interval(2,ExecutionPhase::Compute,105,125,8);check(!m.snapshot().phases[5]->gpu_nanoseconds && m.snapshot().phases[5]->gpu_samples==1);
    m.record_interval(3,ExecutionPhase::Upload,10,20,4);m.record_interval(4,ExecutionPhase::Upload,20,40,6);check(m.snapshot().phases[4]->gpu_nanoseconds==10);
    m.record_interval(5,ExecutionPhase::Upload,0,0);check(!m.snapshot().phases[4]->gpu_nanoseconds);
    fails([&]{m.record_io(1,1,2,3);});fails([&]{m.record_interval(7,ExecutionPhase::Wait,20,10);});
    m.record_io(7,0,0,0);check(m.snapshot().submissions==0);m.record_io(8,2,30,40);check(m.snapshot().upload_bytes==30 && m.snapshot().download_bytes==40);
    ExecutionMetrics submission_only;submission_only.record_submissions(1,7);
    auto submission_snapshot=submission_only.snapshot();check(submission_snapshot.submissions==7);
    check(!submission_snapshot.upload_bytes && !submission_snapshot.download_bytes);
    fails([&]{submission_only.record_submissions(1,1);});
    fails([&]{m.record_interval(9,static_cast<ExecutionPhase>(99),0,1);});
    // Overflow must preserve earlier totals and permit a corrected retry with the same event ID.
    m.record_io(9,std::numeric_limits<std::uint64_t>::max()-2,0,0);
    fails([&]{m.record_io(10,1,7,9);});check(m.snapshot().upload_bytes==30);m.record_io(10,0,1,1);
    ExecutionMetrics overflow;auto huge=overflow.register_allocator(1,1,AllocationRole::Other,AllocationDomain::LogicalRequest,"synthetic overflow");
    overflow.allocate(huge,1,std::numeric_limits<std::uint64_t>::max());fails([&]{overflow.allocate(huge,2,1);});overflow.release(huge,1);overflow.allocate(huge,2,1);
    // Two concurrent callers record exactly once; snapshots synchronize with writes.
    ExecutionMetrics concurrent;auto id=concurrent.register_allocator(1,1,AllocationRole::Staging,AllocationDomain::AllocatorBlock,"concurrent synthetic backing events");
    auto worker=[&](std::uint64_t start) { for(std::uint64_t i=0;i<1000;++i) {auto h=start+i;concurrent.allocate(id,h,1);concurrent.record_io(h,1,2,3);concurrent.release(id,h);check(concurrent.snapshot().allocations[1]->live_bytes<=2);} };
    std::thread t1(worker,1),t2(worker,1001);t1.join();t2.join();s=concurrent.snapshot();
    check(s.allocations[1]->live_bytes==0 && s.allocations[1]->allocations==2000 && s.submissions==2000 && s.upload_bytes==4000);
    std::cout<<"execution metrics CPU contracts passed\n";return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

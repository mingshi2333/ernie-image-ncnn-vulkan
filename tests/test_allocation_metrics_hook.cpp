// SPDX-License-Identifier: MIT
#include "allocation_metrics_hook.h"
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace ernie;
void require(bool b){if(!b)throw std::runtime_error("allocation hook contract");}
// Fake allocator's return and release are independent of observer validity.
struct Backend {
    int frees=0;
    std::uint64_t allocate(bool succeeds,std::uint64_t h,std::uint64_t bytes,VulkanMemoryClass type={}) {
        if(!succeeds)return 0;
        allocation_memory_created(1,2,h,bytes,type);return h;
    }
    void free(std::uint64_t h){allocation_memory_destroyed(1,2,h);++frees;}
};
int main(){try {
    {
        AllocationMeasurementSession s;Backend b;
#if ERNIE_ALLOCATION_HOOKS_ENABLED
        require(s.snapshot().available && !s.snapshot().all_memory);
#else
        require(!s.snapshot().available && !s.snapshot().all_memory);
#endif
        allocation_allocator_created(1,2);allocation_allocator_role(1,2,AllocationRole::Weight);
        require(b.allocate(false,8,10)==0);require(b.allocate(true,8,10,{1,0,1,false})==8);
        require(b.allocate(true,9,20,{2,1,6,true})==9);b.free(8);
        require(b.allocate(true,10,15,{1,0,1,false})==10);
#if ERNIE_ALLOCATION_HOOKS_ENABLED
        auto r=s.snapshot();require(r.valid && r.all_memory->live_bytes==35 && r.all_memory->peak_bytes==35);
        require(r.by_device_memory_class.at({1,{2,1,6,true}}).live_bytes==20);
#endif
        b.free(9);b.free(10);allocation_allocator_destroyed(1,2);require(b.frees==3);
        allocation_allocator_created(1,2);b.allocate(true,8,1);b.free(8);allocation_allocator_destroyed(1,2);
        require(s.snapshot().valid);
    }
#if ERNIE_ALLOCATION_HOOKS_ENABLED
    {
        AllocationMeasurementSession s;Backend b;allocation_allocator_created(1,2);
        b.allocate(true,8,10);b.free(8);b.free(8);
        require(!s.snapshot().valid && b.frees==2);
        // Broken accounting cannot stop subsequent allocation or free.
        require(b.allocate(true,9,12)==9);b.free(9);require(b.frees==3);
    }
    { AllocationMeasurementSession s;allocation_memory_destroyed(1,2,9);require(!s.snapshot().valid); }
    { AllocationMeasurementSession s;allocation_allocator_created(1,2);allocation_memory_created(1,2,9,5,{});allocation_allocator_destroyed(1,2);require(!s.snapshot().valid); }
    {
        AllocationMeasurementSession s;allocation_allocator_created(1,2);
        auto run=[](std::uint64_t base){for(std::uint64_t i=0;i<1000;++i){allocation_memory_created(1,2,base+i,7,{});allocation_memory_destroyed(1,2,base+i);}};
        std::thread a(run,1),b(run,1001);a.join();b.join();auto r=s.snapshot();require(r.valid && r.all_memory->live_bytes==0 && r.all_memory->allocations==2000);
        allocation_allocator_destroyed(1,2);
    }
#endif
    std::cout<<"allocation hooks contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

// SPDX-License-Identifier: MIT
// Actual Vulkan allocation contract. Never loads a model package.
#include "net.h"
#include "command.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
#include "allocation_metrics_hook.h"
#endif
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
constexpr const char* graph="7767517\n2 2\nInput input 0 1 in\nBinaryOp double 1 1 in out 0=2 1=1 2=2.0\n";
void cpu_preflight() {
    // Text params preserve integer vs floating storage; 2=2 is not float 2.
    ncnn::Net net;net.opt.use_vulkan_compute=false;net.opt.num_threads=2;
    require(net.load_param_mem(graph)==0,"CPU graph preflight load failed");
    const unsigned char empty_weights[4]={0,0,0,0};
    require(net.load_model(empty_weights)==0,"CPU graph preflight weights differ");
    ncnn::Mat input(256),output;
    for(int i=0;i<256;i++)input[i]=(i-128)*.125f;
    auto ex=net.create_extractor();
    require(ex.input("in",input)==0 && ex.extract("out",output)==0,"CPU graph preflight failed");
    for(int i=0;i<256;i++)require(output[i]==(i-128)*.25f,"CPU scalar parameter/oracle differs");
}
struct Instance {
    Instance() { ncnn::create_gpu_instance(); }
    ~Instance() { ncnn::destroy_gpu_instance(); }
};
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
using Session=ernie::AllocationMeasurementSession;
std::uint64_t live(Session& s) {
    const auto x=s.snapshot();
    require(x.available && x.valid && x.all_memory.has_value(), "unavailable/invalid actual allocation measurement");
    return x.all_memory->live_bytes;
}
void snapshot(Session& s, std::ostream& out, const char* phase) {
    const auto x=s.snapshot();
    require(x.available && x.valid,"invalid allocation session");
    out << "{\"phase\":\"" << phase << "\",\"available\":true,\"valid\":true,\"live_bytes\":";
    if(x.all_memory) out << x.all_memory->live_bytes; else out << "null";
    out << ",\"peak_bytes\":";
    if(x.all_memory) out << x.all_memory->peak_bytes; else out << "null";
    out << ",\"allocations\":";
    if(x.all_memory) out << x.all_memory->allocations; else out << "null";
    out << ",\"memory_classes\":[";
    bool first=true;
    for(const auto& item:x.by_device_memory_class) {
        if(!first)out << ','; first=false;
        const auto& c=item.first.second; const auto& t=item.second;
        out << "{\"device\":" << item.first.first << ",\"type_index\":" << c.type_index << ",\"heap_index\":" << c.heap_index
            << ",\"property_flags\":" << c.property_flags << ",\"heap_flags\":" << c.heap_flags
            << ",\"host_import\":" << (c.imported_host?"true":"false")
            << ",\"live_bytes\":" << t.live_bytes << ",\"peak_bytes\":" << t.peak_bytes
            << ",\"allocations\":" << t.allocations << '}';
    }
    out << "],\"device_identities\":[";
    first=true;
    for(const auto& item:x.devices) {
        if(!first)out << ','; first=false;
        const auto& d=item.second;
        require(d.index>=0 && !d.name.empty(),"missing actual device identity");
        out << "{\"handle\":" << item.first << ",\"index\":" << d.index
            << ",\"vendor_id\":" << d.vendor_id << ",\"device_id\":" << d.device_id
            << ",\"api_version\":" << d.api_version << ",\"driver_version\":" << d.driver_version
            << ",\"name_utf8_hex\":\"";
        for(unsigned char c:d.name)out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
        out << "\",\"pipeline_cache_uuid\":\"";
        for(auto c:d.pipeline_cache_uuid)out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
        out << std::dec << "\"}";
    }
    for(const auto& item:x.allocator_metrics.allocators)
        require(x.devices.count(item.first.device)==1,"allocator device identity unobserved");
    out << "],\"allocators\":[";
    first=true;
    for(const auto& item:x.allocator_metrics.allocators) {
        if(!first)out << ','; first=false;
        const auto& id=item.first;const auto& a=item.second;
        out << "{\"device\":" << id.device << ",\"allocator\":" << id.allocator
            << ",\"generation\":" << id.generation << ",\"role_enum\":" << static_cast<int>(a.role)
            << ",\"active\":" << (a.active?"true":"false") << ",\"live_bytes\":" << a.totals.live_bytes
            << ",\"peak_bytes\":" << a.totals.peak_bytes << ",\"allocations\":" << a.totals.allocations << '}';
    }
    out << "]}\n";
}
#else
struct Session {};
void snapshot(Session&, std::ostream& out, const char* phase) {
    out << "{\"phase\":\"" << phase << "\",\"available\":false,\"valid\":true,\"live_bytes\":null,\"peak_bytes\":null,\"allocations\":null}\n";
}
#endif
}
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--cpu-preflight") {
        try {cpu_preflight();return 0;} catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
    }
    if(argc!=3) { std::cerr << "usage: ernie-allocation-vulkan-contract OUTPUT.f32 EVENTS.jsonl\n";return 2; }
    try {
        std::ofstream events(argv[2]); require(bool(events),"open events failed");
        cpu_preflight();
        Session session; // Must precede instance/device construction, including dummy allocators.
        snapshot(session,events,"before_instance");
        ncnn::Mat output;
        {
            Instance instance;
            if(ncnn::get_gpu_count()<1) return 77;
            const auto* device=ncnn::get_gpu_device(ncnn::get_default_gpu_index());
            require(device!=nullptr,"device missing");
            snapshot(session,events,"device_created");
            {
                ncnn::VkBlobAllocator pool(device,1024*1024);
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
                const auto baseline=live(session);
#endif
                auto* first=pool.fastMalloc(4096); require(first!=nullptr,"blob allocation failed");
                const auto memory=first->memory; const auto offset=first->offset;
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
                const auto allocated=live(session); require(allocated>baseline,"real blob allocation not observed");
#endif
                snapshot(session,events,"blob_allocated");
                pool.fastFree(first);
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
                require(live(session)==allocated,"logical free incorrectly removed physical pool reservation");
#endif
                auto* second=pool.fastMalloc(4096); require(second!=nullptr,"blob reuse failed");
                require(second->memory==memory && second->offset==offset,"expected same physical pool region reuse");
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
                require(live(session)==allocated,"pool reuse changed physical live bytes");
#endif
                pool.fastFree(second); snapshot(session,events,"blob_reused_and_returned");
                pool.clear();
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
                require(live(session)==baseline,"pool clear did not release exact physical allocation");
#endif
                snapshot(session,events,"blob_cleared");
            }
            {
                ncnn::VkBlobAllocator images(device,1024*1024);
                auto* image=images.fastMalloc(4,4,1,4u,1);
                require(image!=nullptr,"small image allocation failed");
                snapshot(session,events,"image_live");
                images.fastFree(image);images.clear();
            }
            {
                ncnn::VkBlobAllocator cache(device,1024*1024);
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
                ernie::allocation_allocator_role((std::uint64_t)device->vkdevice(),(std::uint64_t)&cache,ernie::AllocationRole::Cache);
#endif
                auto* c=cache.fastMalloc(4096);require(c!=nullptr,"cache allocation failed");
                snapshot(session,events,"cache_live");cache.fastFree(c);cache.clear();
            }
            {
                ncnn::VkWeightAllocator weight(device,false,1024*1024);
                ncnn::VkWeightAllocator host_weight(device,true,1024*1024);
                ncnn::VkStagingAllocator staging(device);
                auto* w=weight.fastMalloc(4096);require(w!=nullptr,"weight allocation failed");
                auto* h=host_weight.fastMalloc(4096);require(h!=nullptr,"host preferred weight allocation failed");
                auto* s=staging.fastMalloc(4096);require(s!=nullptr,"staging allocation failed");
                snapshot(session,events,"weight_host_preferred_staging_live");
                weight.fastFree(w);host_weight.fastFree(h);staging.fastFree(s);
                weight.clear();host_weight.clear();staging.clear();
            }
            {
                ncnn::Net net;
                net.opt.num_threads=2;net.opt.use_vulkan_compute=true;
                net.opt.use_fp16_storage=net.opt.use_fp16_packed=net.opt.use_fp16_arithmetic=false;
                net.opt.use_bf16_storage=net.opt.use_bf16_packed=false;
                net.opt.use_packing_layout=false;
                net.set_vulkan_device(device);
                require(net.load_param_mem(graph)==0,"graph load failed");
                const unsigned char empty_weights[4]={0,0,0,0};
                require(net.load_model(empty_weights)==0,"unexpected graph weight bytes");
                ncnn::VkBlobAllocator blobs(device,1024*1024);
                ncnn::VkStagingAllocator staging(device);
                ncnn::Option opt=net.opt;
                opt.blob_vkallocator=opt.workspace_vkallocator=&blobs;opt.staging_vkallocator=&staging;
                ncnn::Mat input(256);for(int i=0;i<256;i++)input[i]=(i-128)*0.125f;
                ncnn::VkMat gi,go;
                ncnn::VkCompute command(device);
                command.record_upload(input,gi,opt);
                auto ex=net.create_extractor();ex.set_blob_vkallocator(&blobs);ex.set_workspace_vkallocator(&blobs);ex.set_staging_vkallocator(&staging);
                require(ex.input("in",gi)==0,"input failed");require(ex.extract("out",go,command)==0,"extract failed");
                command.record_download(go,output,opt);require(command.submit_and_wait()==0,"GPU submit failed");
                require(output.w==256 && output.elemsize==4u && output.elempack==1,"output layout differs");
                for(int i=0;i<256;i++)require(output[i]==(i-128)*0.25f,"GPU numerical output differs from exact FP32 oracle");
                snapshot(session,events,"gpu_output_complete");
            }
            snapshot(session,events,"user_allocators_destroyed");
        }
        snapshot(session,events,"instance_destroyed");
#ifdef ERNIE_REAL_ALLOCATION_HOOKS
        require(live(session)==0,"physical Vulkan memory still live after complete cleanup");
        const auto final=session.snapshot();
        bool roles[4]={false,false,false,false};
        for(const auto& item:final.allocator_metrics.allocators) {
            require(!item.second.active && item.second.live.empty(),"allocator still active after cleanup");
            const int role=static_cast<int>(item.second.role);
            if(role>=0 && role<4 && item.second.totals.allocations>0)roles[role]=true;
        }
        for(bool seen:roles)require(seen,"expected weight/blob/staging/cache physical allocation not observed");
#endif
        std::ofstream data(argv[1],std::ios::binary);require(bool(data),"open output failed");
        data.write(static_cast<const char*>(output.data),256*sizeof(float));data.close();events.close();
        require(bool(data)&&bool(events),"output close failed");
        std::cout << "actual allocator reuse/free and exact GPU output passed\n";return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}

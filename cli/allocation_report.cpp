// SPDX-License-Identifier: MIT
#include "allocation_report.h"
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
namespace ernie::cli {
namespace {
std::string quoted(const std::string& s) {
    std::ostringstream out;out << '"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\')out << '\\' << char(c);
        else if(c<32)out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
    out << '"';return out.str();
}
const char* boolean(bool b){return b?"true":"false";}
const char* role(AllocationRole r) {
    switch(r){case AllocationRole::Weight:return "weight";case AllocationRole::Blob:return "blob";
        case AllocationRole::Staging:return "staging";case AllocationRole::Cache:return "cache";default:return "other";}
}
void totals(std::ostream& out,const AllocationTotals& t) {
    out << "{\"live_bytes\":" << t.live_bytes << ",\"peak_bytes\":" << t.peak_bytes << ",\"allocations\":" << t.allocations << '}';
}
std::string hex_bytes(const std::array<std::uint8_t,16>& data) {
    std::ostringstream out;for(auto b:data)out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);return out.str();
}
void write_new(const std::filesystem::path& path,const std::string& data) {
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0)throw std::system_error(errno,std::generic_category(),"Cannot create allocation report");
    size_t offset=0;int failure=0;
    while(offset<data.size()) {
        const auto n=::write(fd,data.data()+offset,data.size()-offset);
        if(n<0 && errno==EINTR)continue;
        if(n<=0){failure=n<0?errno:EIO;break;}
        offset+=size_t(n);
    }
    if(::close(fd)<0 && !failure)failure=errno;
    if(failure)throw std::system_error(failure,std::generic_category(),"Cannot finish allocation report");
}
}
std::string allocation_report_json(const AllocationHookSnapshot& s,bool initial,bool final,bool trace,
    const char* status,const std::string& error,std::uint64_t ns) {
    bool released=s.all_memory && s.all_memory->live_bytes==0;
    bool identities=true;
    for(const auto& item:s.allocator_metrics.allocators) {
        if(item.second.active || !item.second.live.empty())released=false;
        if(!s.devices.count(item.first.device))identities=false;
    }
    const bool complete=s.available && s.valid && released && identities && !initial && !final;
    std::ostringstream out;
    out << "{\"schema_version\":1,\"measurement_mode\":\"allocation_diagnostic\","
        << "\"allocation_domain\":\"vk_device_memory\",\"coverage_scope\":\"observed_ncnn_allocator_lifetime\","
        << "\"concurrency_policy\":\"standalone_single_generation_process\","
        << "\"formal_speed_eligible\":false,\"formal_memory_eligible\":false,"
        << "\"identity_status\":\"requires_frozen_parent_binding\","
        << "\"trace_enabled\":" << boolean(trace) << ",\"available\":" << boolean(s.available)
        << ",\"valid\":" << boolean(s.valid) << ",\"coverage_complete\":" << boolean(complete)
        << ",\"initial_instance_present\":" << boolean(initial) << ",\"final_instance_present\":" << boolean(final)
        << ",\"device_identity_complete\":" << boolean(identities)
        << ",\"run_status\":" << quoted(status) << ",\"primary_error\":" << quoted(error)
        << ",\"host_time_scope\":\"cli_generation_and_image_write\",\"host_nanoseconds\":" << ns
        << ",\"cpu_rss\":null,\"gpu_time\":null,\"stage_times\":null,\"total\":";
    if(s.available && s.all_memory)totals(out,*s.all_memory);else out << "null";
    out << ",\"devices\":[";bool first=true;
    for(const auto& item:s.devices) {
        if(!first)out << ',';first=false;const auto& d=item.second;
        out << "{\"process_device_handle\":" << quoted(std::to_string(item.first)) << ",\"index\":" << d.index
            << ",\"vendor_id\":" << d.vendor_id << ",\"device_id\":" << d.device_id
            << ",\"api_version\":" << d.api_version << ",\"driver_version\":" << d.driver_version
            << ",\"name\":" << quoted(d.name) << ",\"pipeline_cache_uuid\":" << quoted(hex_bytes(d.pipeline_cache_uuid)) << '}';
    }
    out << "],\"by_device_memory_class\":[";first=true;
    for(const auto& item:s.by_device_memory_class) {
        if(!first)out << ',';first=false;const auto& c=item.first.second;
        out << "{\"process_device_handle\":" << quoted(std::to_string(item.first.first))
            << ",\"memory_type_index\":" << c.type_index << ",\"heap_index\":" << c.heap_index
            << ",\"property_flags\":" << c.property_flags << ",\"heap_flags\":" << c.heap_flags
            << ",\"host_import\":" << boolean(c.imported_host) << ",\"totals\":";totals(out,item.second);out << '}';
    }
    out << "],\"allocators\":[";first=true;
    for(const auto& item:s.allocator_metrics.allocators) {
        if(!first)out << ',';first=false;const auto& id=item.first;const auto& a=item.second;
        out << "{\"process_device_handle\":" << quoted(std::to_string(id.device))
            << ",\"allocator_handle\":" << quoted(std::to_string(id.allocator)) << ",\"generation\":" << id.generation
            << ",\"role\":" << quoted(role(a.role)) << ",\"active\":" << boolean(a.active)
            << ",\"live_handles\":" << a.live.size() << ",\"totals\":";totals(out,a.totals);out << '}';
    }
    out << "]}\n";return out.str();
}
AllocationReport::AllocationReport(const std::filesystem::path& p,bool trace,bool initial):
    path_(p),trace_(trace),initial_instance_(initial),start_(std::chrono::steady_clock::now()) {
    if(path_.empty() || std::filesystem::exists(std::filesystem::symlink_status(path_)))
        throw std::invalid_argument("Use a new allocation report path");
    const auto parent=path_.has_parent_path()?path_.parent_path():std::filesystem::path(".");
    if(!std::filesystem::is_directory(parent))throw std::invalid_argument("Allocation report parent directory is missing");
}
void AllocationReport::finish(const char* status,const std::string& error,bool final) {
    if(attempted_)throw std::logic_error("Allocation report already attempted");
    attempted_=true;
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start_).count();
    write_new(path_,allocation_report_json(session_.snapshot(),initial_instance_,final,trace_,status,error,std::uint64_t(ns)));
}
}

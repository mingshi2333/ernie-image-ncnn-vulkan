// SPDX-License-Identifier: MIT
#pragma once
#include "allocation_metrics_hook.h"
#include <chrono>
#include <filesystem>
#include <string>
namespace ernie::cli {
// Private CLI adapter. Never owns or destroys a Vulkan context.
class AllocationReport {
public:
    AllocationReport(const std::filesystem::path& path,bool trace,bool initial_instance_present);
    bool attempted() const noexcept { return attempted_; }
    void finish(const char* status,const std::string& primary_error,bool final_instance_present);
private:
    std::filesystem::path path_;
    bool trace_,initial_instance_,attempted_=false;
    std::chrono::steady_clock::time_point start_;
    AllocationMeasurementSession session_;
};
std::string allocation_report_json(const AllocationHookSnapshot&,bool initial_instance,bool final_instance,
    bool trace,const char* status,const std::string& error,std::uint64_t host_nanoseconds);
}

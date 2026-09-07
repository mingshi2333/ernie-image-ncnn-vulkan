// SPDX-License-Identifier: MIT
#include "generation_report.h"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace ernie::cli {
namespace {
std::string quote(const std::string& value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << '"';
    for (unsigned char c : value)
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    out << '"';
    return out.str();
}
void finite_time(double value)
{
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid generation report time");
}
} // namespace

std::string generation_report_json(const GenerationRequest& r, const GenerationResult& s,
    const std::vector<Progress>& progress, double write_seconds, bool instrumented)
{
    finite_time(write_seconds); finite_time(s.elapsed_seconds); finite_time(s.vae_seconds);
    finite_time(s.elapsed_seconds + write_seconds); finite_time(s.vae_seconds + write_seconds);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(17)
        << "{\"schema_version\":1,\"status\":\"success\",\"trace_enabled\":" << !r.trace.empty()
        << ",\"allocation_instrumentation\":" << instrumented
        << ",\"shape\":[" << s.image.width << ',' << s.image.height << "],\"shape_order\":\"WH\""
        << ",\"model\":{\"schema_version\":" << s.model_schema
        << ",\"source_width\":" << s.source_width << ",\"source_height\":" << s.source_height
        << ",\"text_bucket\":" << s.text_bucket << ",\"dit_text_tokens\":" << s.dit_text_tokens << '}'
        << ",\"vulkan_gpu_index\":" << s.vulkan_gpu_index
        << ",\"request\":{\"device\":" << quote(r.device) << ",\"precision\":" << quote(r.precision)
        << ",\"vae_device\":" << quote(r.vae_device) << ",\"vae_convolution\":" << quote(r.vae_convolution)
        << ",\"text_device\":" << quote(r.text_device) << ",\"text_down_vector\":" << r.text_down_vector
        << ",\"threads\":" << r.threads << ",\"steps\":" << r.steps << ",\"seed\":" << r.seed
        << ",\"dit_weights\":" << quote(r.dit_weights) << ",\"gpu_reserve_mib\":" << r.gpu_reserve_mib
        << ",\"dit_cache_mib\":" << r.dit_cache_mib << ",\"ram_reserve_mib\":" << r.ram_reserve_mib
        << ",\"model_loading\":" << quote(r.model_loading)
        << ",\"img2img\":" << r.input_image.has_value() << ",\"resize\":" << quote(r.input_resize)
        << ",\"strength\":";
    if (r.input_image) out << r.strength; else out << "null";
    out << '}'
        << ",\"model_loading_requested\":" << quote(s.mapped_model_loading_requested ? "mapped" : "stdio")
        << ",\"placement_requests\":{\"gpu\":" << s.device_weight_requests << ",\"ram\":" << s.host_weight_requests
        << ",\"budget_unavailable\":" << s.unavailable_memory_queries << '}'
        << ",\"weight_cache\":{\"hits\":" << s.weight_cache_hits << ",\"loads\":" << s.weight_cache_loads
        << ",\"peak_charged_bytes\":" << s.weight_cache_peak_bytes << ",\"peak_nets\":" << s.weight_cache_peak_nets
        << ",\"evictions\":" << s.weight_cache_evictions << ",\"budget_unavailable\":" << s.unavailable_host_memory_queries << '}'
        << ",\"prompt\":" << quote(s.prompt) << ",\"token_ids\":[";
    for (size_t i = 0; i < s.token_ids.size(); ++i) { if (i) out << ','; out << s.token_ids[i]; }
    out << "],\"pe\":{\"enabled\":" << !r.pe_model.empty() << ",\"greedy\":" << r.pe.greedy
        << ",\"max_tokens\":" << r.pe.max_tokens << ",\"generated_tokens\":" << s.pe_generated_tokens
        << ",\"eos\":" << s.pe_eos << '}'
        << ",\"generation_seconds\":" << s.elapsed_seconds << ",\"image_write_seconds\":" << write_seconds
        << ",\"vae_and_image_encode_seconds\":" << s.vae_seconds + write_seconds
        << ",\"total_seconds\":" << s.elapsed_seconds + write_seconds
        << ",\"progress\":[";
    for (size_t i = 0; i < progress.size(); ++i)
    {
        const auto& p = progress[i]; finite_time(p.seconds);
        if (i) out << ',';
        out << "{\"stage\":" << quote(p.stage) << ",\"current\":" << p.current
            << ",\"total\":" << p.total << ",\"seconds\":" << p.seconds << '}';
    }
    out << "],\"time_scope\":\"host generation and image write; excludes report serialization and process startup\","
        << "\"progress_scope\":\"verify is cumulative from generation start; text is the text stage; denoise is per step\","
        << "\"memory_scope\":\"placement requests and charged cache weights; not complete residency or process peak\"}\n";
    return out.str();
}

void write_generation_report(const std::filesystem::path& path, const GenerationRequest& r,
    const GenerationResult& s, const std::vector<Progress>& progress, double seconds, bool instrumented)
{
    const auto data = generation_report_json(r, s, progress, seconds, instrumented);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
#ifdef _WIN32
    auto* file = _wfopen(path.c_str(), L"wbx");
#else
    auto* file = std::fopen(path.c_str(), "wbx");
#endif
    if (!file) throw std::system_error(errno, std::generic_category(), "Cannot create generation report");
    int failure = 0;
    if (std::fwrite(data.data(), 1, data.size(), file) != data.size()) failure = errno ? errno : EIO;
    if (std::fclose(file) && !failure) failure = errno ? errno : EIO;
    if (failure) throw std::system_error(failure, std::generic_category(), "Cannot finish generation report");
}
} // namespace ernie::cli

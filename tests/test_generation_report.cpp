// SPDX-License-Identifier: MIT
#include "generation_report.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() /
        fs::u8path(u8"ernie-generation-report-\u62a5\u544a \U0001f5bc-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        ernie::GenerationRequest request;
        ernie::GenerationResult result;
        if (result.model_schema || result.text_bucket || result.dit_text_tokens ||
            result.source_width || result.source_height || result.vulkan_gpu_index != -1 ||
            result.prefetch_started || result.prefetch_used || result.prefetch_skipped ||
            result.prefetch_peak_charged_bytes || result.prefetch_overlap_seconds ||
            result.gpu_device_allocations || result.gpu_host_allocations || result.gpu_host_peak_bytes ||
            result.gpu_memory_fallbacks || result.gpu_allocation_failures ||
            result.gpu_host_device_local_allocations || result.gpu_host_non_device_local_allocations ||
            result.memory_retries || result.attention_query_rows != 128)
            throw std::runtime_error("Unresolved public result defaults changed");
        result.image.width = 512; result.image.height = 384;
        result.prompt = "中文 \"quote\"\\\r\n\t";
        result.prompt.push_back('\0');
        result.token_ids = {1, 23, 131071};
        result.model_schema = 3; result.text_bucket = 32; result.dit_text_tokens = 64;
        result.source_width = result.source_height = 1024;
        result.vulkan_gpu_index = 0; result.elapsed_seconds = 1.25; result.vae_seconds = .75;
        result.host_weight_requests = 262; result.weight_cache_hits = 42;
        result.weight_cache_peak_bytes = 5687678976ull;
        result.mapped_model_loading_requested = true;
        request.precision = "fp32"; request.dit_prefetch_mib = 1024;
        result.prefetch_started = 3; result.prefetch_used = 2; result.prefetch_skipped = 1;
        result.prefetch_peak_charged_bytes = 536870912; result.prefetch_overlap_seconds = .25;
        result.gpu_device_allocations = 32; result.gpu_host_allocations = 16;
        result.gpu_host_peak_bytes = 536870912; result.gpu_memory_fallbacks = 8;
        result.gpu_allocation_failures = 2; result.gpu_host_device_local_allocations = 4;
        result.gpu_host_non_device_local_allocations = 12;
        result.memory_retries = 2; result.attention_query_rows = 32;
        std::vector<ernie::Progress> progress{{"text", 3, 32, .5}, {"denoise", 1, 1, .125}};
        const auto data = ernie::cli::generation_report_json(request, result, progress, .125, false);
        ernie::cli::write_generation_report(path, request, result, progress, .125, false);
        bool rejected = false;
        result.prompt = "Replacement content must not reach the existing report";
        try { ernie::cli::write_generation_report(path, request, result, progress, .125, false); }
        catch (const std::system_error&) { rejected = true; }
        std::ifstream input(path, std::ios::binary);
        const std::string saved((std::istreambuf_iterator<char>(input)), {});
        input.close();
        if (!rejected || saved != data) throw std::runtime_error("Generation report overwrote an existing file");
        for (double invalid : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            rejected = false;
            try { ernie::cli::generation_report_json(request, result, progress, invalid, false); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid report duration was accepted");
            result.prefetch_overlap_seconds = invalid;
            rejected = false;
            try { ernie::cli::generation_report_json(request, result, progress, .125, false); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid prefetch duration was accepted");
            result.prefetch_overlap_seconds = .25;
        }
        fs::remove(path);
        std::cout << data;
    }
    catch (const std::exception& error)
    {
        std::error_code ignored; fs::remove(path, ignored);
        std::cerr << error.what() << '\n'; return 1;
    }
}

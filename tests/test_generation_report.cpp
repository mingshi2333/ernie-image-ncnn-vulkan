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
            result.source_width || result.source_height || result.vulkan_gpu_index != -1)
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

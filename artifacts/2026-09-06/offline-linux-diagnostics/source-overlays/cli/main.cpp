// SPDX-License-Identifier: MIT
#include "image_io.h"
#include "options.h"
#include <chrono>
#include <iostream>
#ifdef ERNIE_CLI_ALLOCATION_METRICS
#include "allocation_report.h"
#include <gpu.h>
#include <memory>
#endif

int main(int argc, char **argv)
{
#ifdef ERNIE_CLI_ALLOCATION_METRICS
    std::unique_ptr<ernie::cli::AllocationReport> metrics;
    const char* failure_phase="input_preparation_failed";
#endif
    try
    {
        const auto options = ernie::cli::parse_options(argc, argv);
        auto request = options.generation;
        if (options.help)
        {
            std::cout << ernie::cli::usage();
            return 0;
        }
        if (options.diagnose_only)
        {
            const auto info = ernie::diagnose(request.model);
            std::cout << "vulkan_compiled=" << (info.vulkan_compiled ? "true" : "false") << '\n'
                      << "gpu_count=" << info.vulkan_devices.size() << '\n'
                      << "default_gpu_index=" << info.default_gpu_index << '\n';
            if (!info.vulkan_error.empty())
                std::cout << "vulkan_error=" << info.vulkan_error << '\n';
            for (const auto &device : info.vulkan_devices)
                std::cout << "gpu[" << device.index << "].name=" << device.name << '\n'
                          << "gpu[" << device.index
                          << "].fp16_storage=" << (device.fp16_storage ? "true" : "false") << '\n'
                          << "gpu[" << device.index
                          << "].bf16_storage=" << (device.bf16_storage ? "true" : "false") << '\n';
            if (info.model_config_loaded)
                std::cout << "model_config_schema=" << info.model_config_schema << '\n'
                          << "packed_width=" << info.packed_width << '\n'
                          << "packed_height=" << info.packed_height << '\n'
                          << "text_bucket=" << info.text_bucket << '\n'
                          << "dit_text_tokens=" << info.dit_text_tokens << '\n'
                          << "text_layers=" << info.text_layers << '\n'
                          << "dit_layers=" << info.dit_layers << '\n';
            return 0;
        }
        if (options.verify_only)
        {
            if (!request.model.empty())
                ernie::verify_model(request.model);
            if (!request.pe_model.empty())
                ernie::verify_pe_model(request.pe_model);
            std::cout << "Model verified\n";
            return 0;
        }
#ifdef ERNIE_CLI_ALLOCATION_METRICS
        std::cerr << "Allocation instrumentation enabled; timings are diagnostic, not formal speed measurements\n";
        if (!options.metrics_json.empty())
            metrics=std::make_unique<ernie::cli::AllocationReport>(options.metrics_json,!request.trace.empty(),ncnn::get_gpu_instance()!=VK_NULL_HANDLE);
#endif
        if (!options.input.empty())
        {
            auto image = ernie::cli::read_image(options.input, options.background);
            request.input_source_width = image.width;
            request.input_source_height = image.height;
            request.input_alpha_background = options.background;
            if (!options.resize.empty())
            {
                const std::array<uint8_t, 3> fill = options.background_explicit
                                                        ? options.background
                                                        : std::array<uint8_t, 3>{0, 0, 0};
                image = ernie::cli::resize_image(image, request.width, request.height, options.resize, fill);
                request.input_resize = options.resize;
                request.input_resize_background = fill;
            }
            request.input_image = std::move(image);
        }
#ifdef ERNIE_CLI_ALLOCATION_METRICS
        failure_phase="generation_failed";
#endif
        const auto result = ernie::generate(
            request,
            [&](const ernie::Progress &p)
            {
                if (p.stage == "verify")
                    std::cout << "Model verified: " << p.seconds << " s";
                else if (p.stage == "text")
                    std::cout << "Text conditioned: " << p.current << " tokens, " << p.seconds
                              << " s, embeddings " << (request.embeddings.empty() ? "computed" : "loaded");
                else if (p.stage == "denoise")
                    std::cout << "Denoise " << p.current << '/' << p.total << ": " << p.seconds << " s";
                else
                    std::cout << p.stage << ' ' << p.current << '/' << p.total;
                std::cout << std::endl;
            });
        const auto write_start = std::chrono::steady_clock::now();
#ifdef ERNIE_CLI_ALLOCATION_METRICS
        failure_phase="image_write_failed";
#endif
        ernie::cli::write_image(options.output, result.image);
        const auto write_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - write_start).count();
#ifdef ERNIE_CLI_ALLOCATION_METRICS
        failure_phase="report_write_failed";
        if (metrics)metrics->finish("success","",ncnn::get_gpu_instance()!=VK_NULL_HANDLE);
#endif
        if (!request.pe_model.empty())
            std::cout << "PE generated " << result.pe_generated_tokens << " tokens, "
                      << (result.pe_eos ? "EOS reached" : "token limit reached")
                      << "\nEnhanced prompt: " << result.prompt << '\n';
        std::cout << "VAE and image encode: " << result.vae_seconds + write_seconds << " s\n"
                  << "Saved " << result.image.width << 'x' << result.image.height
                  << " image: " << options.output.string() << '\n'
                  << "Total: " << result.elapsed_seconds + write_seconds << " s\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
#ifdef ERNIE_CLI_ALLOCATION_METRICS
        if (metrics && !metrics->attempted())
        {
            try {metrics->finish(failure_phase,error.what(),ncnn::get_gpu_instance()!=VK_NULL_HANDLE);}
            catch(const std::exception& report_error) {std::cerr << "Allocation report also failed: " << report_error.what() << '\n';}
        }
#endif
        return 1;
    }
}

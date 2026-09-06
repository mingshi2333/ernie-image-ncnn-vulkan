// SPDX-License-Identifier: MIT
#include "gpu_context.h"
#include <ernie/pipeline.h>
#include <platform.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#include <future>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
    try
    {
#if NCNN_VULKAN
        if (argc == 2 && std::string(argv[1]) == "--without-driver")
        {
            // The parent process selects an absent ICD; do not mutate process
            // environment while another context could own ncnn's global state.
            for (int attempt = 0; attempt < 2; ++attempt)
            {
                const auto info = ernie::diagnose();
                if (!info.vulkan_compiled || !info.vulkan_devices.empty() ||
                    info.default_gpu_index != -1 || info.vulkan_error != "Cannot initialize Vulkan" ||
                    ncnn::get_gpu_instance() != VK_NULL_HANDLE)
                    throw std::runtime_error("Missing driver diagnosis must return a reason without owning Vulkan");
                bool rejected = false;
                try { ernie::GpuContext required(true); }
                catch (const std::runtime_error &error)
                { rejected = std::string(error.what()) == "Cannot initialize Vulkan"; }
                if (!rejected || ncnn::get_gpu_instance() != VK_NULL_HANDLE)
                    throw std::runtime_error("Missing driver must still reject inference without leaking state");
            }
            std::cout << "Missing driver diagnosis and strict inference remain independent\n";
            return 0;
        }
        if (ncnn::create_gpu_instance())
        {
            ncnn::destroy_gpu_instance();
            return 77;
        }
        ncnn::destroy_gpu_instance();
        {
        ernie::GpuContext generation(true, -1, false);
        const int count = ncnn::get_gpu_count();
        const auto instance = ncnn::get_gpu_instance();
        const auto same_thread = ernie::diagnose();
        if (same_thread.vulkan_devices.size() != size_t(count) || ncnn::get_gpu_instance() != instance)
            throw std::runtime_error("Nested diagnosis changed live GPU instance");
        auto first = std::async(std::launch::async, [] { return ernie::diagnose(); });
        auto second = std::async(std::launch::async, [] { return ernie::diagnose(); });
        if (first.get().vulkan_devices.size() != size_t(count) || second.get().vulkan_devices.size() != size_t(count)
            || ncnn::get_gpu_instance() != instance)
            throw std::runtime_error("Concurrent diagnosis destroyed live GPU instance");
        try
        {
            ernie::GpuContext invalid(true, count, true);
            throw std::runtime_error("Unavailable device accepted");
        }
        catch (const std::invalid_argument &) {}
        if (ncnn::get_gpu_instance() != instance)
            throw std::runtime_error("Failed nested context destroyed its owner");
        }
        if (ncnn::get_gpu_instance() != VK_NULL_HANDLE)
            throw std::runtime_error("Final owner leaked its GPU instance");
        if (ncnn::create_gpu_instance())
            throw std::runtime_error("External owner cannot initialize Vulkan");
        const auto external = ncnn::get_gpu_instance();
        {
            ernie::GpuContext borrowed(true, -1, false);
            ernie::diagnose();
        }
        if (ncnn::get_gpu_instance() != external)
            throw std::runtime_error("Library destroyed an external GPU instance");
        ncnn::destroy_gpu_instance();
#else
        (void)argc;
        (void)argv;
        ernie::GpuContext disabled(false);
        if (ernie::diagnose().vulkan_compiled)
            throw std::runtime_error("CPU build reports Vulkan");
#endif
        std::cout << "Shared GPU context survives nested, concurrent and failed diagnostic users\n";
    }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

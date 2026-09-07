// SPDX-License-Identifier: MIT
#include "weight_placement.h"
#include "block_sequence.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
struct Temporary
{
    fs::path path = fs::temp_directory_path() /
        ("ernie-weight-placement-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() { require(fs::create_directory(path), "Cannot create test directory"); }
    ~Temporary() { std::error_code ignored; fs::remove_all(path, ignored); }
};
ernie::ComponentFiles fixture(const fs::path& path)
{
    std::string graph = "7767517\n11 11\n";
    for (int i = 0; i < 10; ++i)
        graph += "Input input" + std::to_string(i) + " 0 1 in" + std::to_string(i) + "\n";
    graph += "Gemm projection 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=1 8=2 9=2 10=-1\n";
    std::ofstream file(path, std::ios::binary);
    const std::uint32_t tag = 0;
    const float weights[] = {2.f, 0.f, 0.f, 3.f};
    file.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    file.write(reinterpret_cast<const char*>(weights), sizeof(weights));
    require(bool(file), "Cannot write weights");
    return {graph, path.string()};
}
void policy_contract(const ernie::ComponentFiles& files)
{
    using namespace ernie;
    auto decision = [](std::uint64_t budget, std::uint64_t usage) {
        return choose_weight_placement(WeightMemory::Auto, false, 100, 20, DeviceMemoryBudget{budget, usage});
    };
    require(!decision(200, 80).host, "Exact budget boundary must fit");
    require(decision(200, 81).host, "Budget usage was not subtracted");
    require(decision(200, 201).available_bytes == 0 && decision(200, 201).host, "Over-budget usage wrapped");
    require(choose_weight_placement(WeightMemory::Auto, false, UINT64_MAX, UINT64_MAX,
        DeviceMemoryBudget{UINT64_MAX, 0}).host, "Required bytes overflowed");
    require(choose_weight_placement(WeightMemory::Auto, true, 1, 0,
        DeviceMemoryBudget{200, 0}).host, "Large shape preference was lost");
    auto absent = choose_weight_placement(WeightMemory::Auto, false, 100, 20, std::nullopt);
    require(!absent.host && !absent.available_bytes && std::string(absent.reason) == "unavailable",
            "Unavailable budget was fabricated");
    require(choose_weight_placement(WeightMemory::Host, false, 1, 0, std::nullopt).host,
            "Explicit RAM mode was ignored");
    require(!choose_weight_placement(WeightMemory::Device, true, 100, 20,
        DeviceMemoryBudget{0, 0}).host, "Explicit GPU mode was ignored");
    int queries = 0;
    WeightPlacement dynamic(WeightMemory::Auto, 20, [&]() -> std::optional<DeviceMemoryBudget> {
        return DeviceMemoryBudget{200, ++queries == 2 ? 190u : 0u};
    });
    require(!dynamic.use_host(files, false) && dynamic.use_host(files, false) &&
            !dynamic.use_host(files, false), "Policy did not re-query changing memory");
    require(queries == 3 && dynamic.host_requests() == 1 && dynamic.device_requests() == 2,
            "Placement counters differ");
    WeightPlacement unavailable(WeightMemory::Auto, 0);
    require(!unavailable.use_host(files, false) && unavailable.unavailable_queries() == 1,
            "Missing reader was not reported");
    WeightPlacement manual(WeightMemory::Host, 0, [&]() -> std::optional<DeviceMemoryBudget> {
        throw std::runtime_error("Manual mode must not need driver budget");
    });
    require(manual.use_host(files, false), "Manual placement failed");
    bool failed = false;
    try { dynamic.use_host({files.param_text, files.weight_path + ".absent"}, false); }
    catch (const fs::filesystem_error&) { failed = true; }
    require(failed && dynamic.device_requests() == 2 && dynamic.host_requests() == 1,
            "Failed weight metadata lookup changed load counters");
}
#if NCNN_VULKAN
int vulkan_contract(const ernie::ComponentFiles& files)
{
    struct Context { ~Context() { ncnn::destroy_gpu_instance(); } } context;
    if (ncnn::create_gpu_instance() || ncnn::get_gpu_count() == 0) return 77;
    const auto* device = ncnn::get_gpu_device();
    ncnn::VkBlobAllocator blobs(device);
    ncnn::VkStagingAllocator staging(device);
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_storage = option.use_fp16_packed = option.use_fp16_arithmetic = false;
    option.use_packing_layout = false;
    option.num_threads = 1;
    option.blob_vkallocator = option.workspace_vkallocator = &blobs;
    option.staging_vkallocator = &staging;
    ncnn::Mat input(2, 1); input[0] = 1.25f; input[1] = -2.f;
    ncnn::Mat constant(1); constant[0] = 0.f;
    ncnn::VkMat gpu_input, gpu_constant;
    {
        ncnn::VkCompute upload(device);
        upload.record_upload(input, gpu_input, option);
        upload.record_upload(constant, gpu_constant, option);
        require(upload.submit_and_wait() == 0, "Fixture upload failed");
    }
    const auto real_reader = ernie::vulkan_memory_budget_reader(device, gpu_input.data->memory_type_index);
    const auto real = real_reader();
    if (real) std::cout << "Vulkan heap budget=" << real->budget_bytes << " usage=" << real->usage_bytes << '\n';
    else std::cout << "Vulkan heap budget unavailable\n";
    int queries = 0;
    ernie::WeightPlacement adaptive(ernie::WeightMemory::Auto, 20,
        [&]() -> std::optional<ernie::DeviceMemoryBudget> {
            return ernie::DeviceMemoryBudget{200, ++queries == 2 ? 190u : 0u};
        });
    ernie::BlockSequenceStats stats;
    const std::vector<ernie::ComponentFiles> models(3, files);
    const std::vector<ncnn::VkMat> constants(9, gpu_constant);
    auto result = ernie::run_block_sequence(models, gpu_input, constants, device, option,
                                          ernie::WeightPolicy::Stream, stats, {}, &adaptive);
    ncnn::Mat host;
    {
        ncnn::VkCompute download(device);
        download.record_download(result, host, option);
        require(download.submit_and_wait() == 0, "Result download failed");
    }
    require(host.w == 2 && host[0] == 10.f && host[1] == -54.f,
            "Alternating GPU/RAM/GPU weight requests changed exact GEMM output");
    require(queries == 3 && adaptive.host_requests() == 1 && adaptive.device_requests() == 2 &&
            stats.load_seconds.size() == 3 && stats.peak_loaded_nets == 1,
            "Stream did not use adaptive placement for every block");
    // A caller can request more headroom without filling the GPU with a stress
    // allocation. Exercise the actual driver reader with that controlled input.
    ernie::WeightPlacement reserve(ernie::WeightMemory::Auto, UINT64_MAX, real_reader);
    require(reserve.use_host(files, false) == bool(real), "Real driver budget selection differs");
    return 0;
}
#endif
} // namespace
int main(int argc, char** argv)
{
    try
    {
        Temporary temp;
        const auto files = fixture(temp.path / "weights.bin");
        if (argc == 2 && std::string(argv[1]) == "vulkan")
        {
#if NCNN_VULKAN
            return vulkan_contract(files);
#else
            return 77;
#endif
        }
        policy_contract(files);
        std::cout << "Adaptive weight budget, overrides, unavailable data and failure contracts passed\n";
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

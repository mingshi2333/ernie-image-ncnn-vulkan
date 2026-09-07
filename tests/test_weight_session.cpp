// SPDX-License-Identifier: MIT
#include "weight_session.h"
#include "block_sequence.h"
#include "layer.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template<class F> void fails(F action)
{
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Expected rejection");
}
int live_nets = 0;
class LifetimeToken : public ncnn::Layer
{
public:
    LifetimeToken() { ++live_nets; }
    ~LifetimeToken() override { --live_nets; }
};
DEFINE_LAYER_CREATOR(LifetimeToken)
std::unique_ptr<ncnn::Net> fake_net()
{
    auto net = std::make_unique<ncnn::Net>();
    require(net->register_custom_layer("LifetimeToken", LifetimeToken_layer_creator) == 0, "register token");
    require(net->load_param_mem("7767517\n1 1\nLifetimeToken token 0 1 out0\n") == 0, "load token");
    return net;
}
void state_contract()
{
    using namespace ernie;
    static_assert(!std::is_copy_constructible<WeightSession::Lease>::value, "Leases must not duplicate ownership");
    const ComponentFiles files{"synthetic graph", "synthetic weights"};
    int request;
    std::uint64_t available = 1000;
    auto inspect = [](const ncnn::Net&) -> std::optional<std::uint64_t> { return 10; };
    auto reader = [&]() -> std::optional<std::uint64_t> { return available; };
    WeightSession session({20, 30}, reader, inspect);
    int requested_host = 0, loaded = 0;
    auto load = [&](bool host) { ++loaded; requested_host += host; return fake_net(); };
    // A repeated sequential scan must hit the admitted prefix even though the
    // sequence exceeds capacity. An LRU cache would yield zero hits here.
    for (int step = 0; step < 3; ++step)
        for (std::size_t block = 0; block < 3; ++block)
        {
            auto lease = session.acquire(block, files, &request, 10, load);
            require(live_nets <= 3 && session.stats().live_bytes <= 20, "Unbounded resident weights");
            lease.complete();
        }
    require(loaded == 5 && requested_host == 2 && session.stats().hits == 4 &&
            session.stats().peak_bytes == 20 && live_nets == 2, "Cache scan thrashed or accounting differs");
    const ComponentFiles changed{"different graph", files.weight_path};
    fails([&] { session.acquire(0, changed, &request, 10, load); });
    int other_request;
    fails([&] { session.acquire(0, files, &other_request, 10, load); });
    {
        auto lease = session.acquire(0, files, &request, 10, load);
        fails([&] { session.acquire(1, files, &request, 10, load); });
        auto moved = std::move(lease);
        fails([&] { lease.net(); });
        session.cancel();
        require(live_nets == 1 && session.stats().live_bytes == 10, "Cancel released an active Net");
        moved.complete();
        require(live_nets == 0 && session.stats().live_bytes == 0, "Cancelled lease leaked");
    }
    fails([&] { session.acquire(0, files, &request, 10, load); });
    {
        WeightSession pressure({20, 30}, reader, inspect);
        auto first = pressure.acquire(0, files, &request, 10, load); first.complete();
        available = 29;
        auto second = pressure.acquire(1, files, &request, 10, load);
        require(!second.cached() && pressure.stats().evictions == 1 && live_nets == 1, "Pressure failed to evict idle weights");
        second.complete();
    }
    available = 1000;
    {
        WeightSession failure({20, 30}, reader, inspect);
        fails([&] { failure.acquire(0, files, &request, 10, [](bool) -> std::unique_ptr<ncnn::Net> { throw std::runtime_error("read failure"); }); });
        auto lease = failure.acquire(0, files, &request, 10, load);
        require(live_nets == 1, "Load failure poisoned session");
        // Abandoned computation is discarded rather than entering the cache.
    }
    require(live_nets == 0, "Abandoned lease leaked");
    {
        WeightSession no_budget({20, 30}, {}, inspect);
        auto lease = no_budget.acquire(0, files, &request, 10, load);
        require(!lease.cached() && no_budget.stats().unavailable_queries, "Missing budget fabricated admission");
        lease.complete();
    }
    {
        WeightSession mismatch({20, 30}, reader, [](const ncnn::Net&) -> std::optional<std::uint64_t> { return 21; });
        auto lease = mismatch.acquire(0, files, &request, 10, load);
        require(!lease.cached() && mismatch.stats().live_bytes == 0, "Measured charge exceeded budget");
    }
    {
        WeightSession fallback({20, 30}, reader, [](const ncnn::Net&) -> std::optional<std::uint64_t> { return std::nullopt; });
        auto lease = fallback.acquire(0, files, &request, 10, load);
        require(!lease.cached(), "Unverified/device allocation entered RAM cache");
    }
    {
        WeightSession::Lease lease;
        {
            WeightSession owner({20, 30}, reader, inspect);
            lease = owner.acquire(0, files, &request, 10, load);
        }
        require(live_nets == 1, "Session destruction invalidated active lease");
        lease.complete();
    }
    require(live_nets == 0, "Session destruction leaked a Net");
    std::cout << "Budget, repeated scan, pressure, identity, failure, move and cancellation contracts passed\n";
}
#if NCNN_VULKAN
int vulkan_contract()
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
    struct Temporary
    {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("ernie-weight-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Temporary() { require(std::filesystem::create_directory(path), "Cannot create test directory"); }
        ~Temporary() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } temp;
    std::string graph = "7767517\n11 11\n";
    for (int i = 0; i < 10; ++i) graph += "Input input" + std::to_string(i) + " 0 1 in" + std::to_string(i) + "\n";
    graph += "Gemm projection 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=1 8=2 9=2 10=-1\n";
    const auto path = temp.path / "weights.bin";
    {
        std::ofstream file(path, std::ios::binary);
        const std::uint32_t tag = 0;
        const float weight[] = {2.f, 0.f, 0.f, 3.f};
        file.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
        file.write(reinterpret_cast<const char*>(weight), sizeof(weight));
        require(bool(file), "Cannot write fixture");
    }
    const std::vector<ernie::ComponentFiles> models(3, {graph, path.string()});
    ncnn::Mat input(2, 1); input[0] = 1.25f; input[1] = -2.f;
    ncnn::Mat constant(1); constant[0] = 0.f;
    ncnn::VkMat gpu_input, gpu_constant;
    {
        ncnn::VkCompute upload(device);
        upload.record_upload(input, gpu_input, option);
        upload.record_upload(constant, gpu_constant, option);
        require(upload.submit_and_wait() == 0, "Upload failed");
    }
    const auto available = ernie::host_memory_available_reader()();
    std::cout << "Host available=" << (available ? std::to_string(*available) : "unavailable") << '\n';
    ernie::WeightSession session({160ull * 1024 * 1024, 0},
        []() -> std::optional<std::uint64_t> { return UINT64_MAX; }, ernie::dit_host_weight_inspector(device));
    ernie::WeightPlacement placement(ernie::WeightMemory::Auto, 0);
    bool has_nonlocal_host = false;
    const auto& memory = device->info.physicalDeviceMemoryProperties();
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        has_nonlocal_host |= (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            !(memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    for (int step = 0; step < 3; ++step)
    {
        ernie::BlockSequenceStats stats;
        const auto result = ernie::run_block_sequence(models, gpu_input, std::vector<ncnn::VkMat>(9, gpu_constant),
            device, option, ernie::WeightPolicy::Stream, stats, {}, &placement, &session);
        ncnn::Mat host;
        ncnn::VkCompute download(device);
        download.record_download(result, host, option);
        require(download.submit_and_wait() == 0 && host.w == 2 && host[0] == 10.f && host[1] == -54.f,
            "Reusing prepared Vulkan weights changed GEMM output");
        require(stats.compute_submissions == 3 && stats.load_seconds.size() == 3, "Repeated execution accounting differs");
    }
    const auto stats = session.stats();
    std::cout << "Vulkan cache hits=" << stats.hits << " loads=" << stats.loads
              << " peak charged bytes=" << stats.peak_bytes << " nets=" << stats.peak_nets << '\n';
    if (has_nonlocal_host)
        require(stats.hits == 4 && stats.loads == 5 && stats.peak_nets == 2, "Vulkan cache was not bounded/reused");
    else require(stats.hits == 0, "Device-local memory was retained as host-only cache");
    require(stats.peak_bytes <= 160ull * 1024 * 1024, "Vulkan cache exceeded budget");
    session.cancel();
    require(session.stats().live_bytes == 0 && session.stats().cached_nets == 0, "Cached weights survived cancellation");
    return 0;
}
#endif
} // namespace
int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "vulkan")
        {
#if NCNN_VULKAN
            return vulkan_contract();
#else
            return 77;
#endif
        }
        state_contract();
        return 0;
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

// SPDX-License-Identifier: MIT
#include "block_sequence.h"
#include "vulkan_denoise.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template<class Error, class Function> std::string expect_failure(Function&& function)
{
    try { function(); }
    catch (const Error& error) { return error.what(); }
    throw std::runtime_error("Expected exception was not raised");
}
bool exact(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (a.empty() || b.empty() || a.dims != b.dims || a.w != b.w || a.h != b.h ||
        a.d != b.d || a.c != b.c || a.elempack != b.elempack || a.elemsize != b.elemsize)
        return false;
    // Ignore allocator alignment padding, which is not part of a tensor.
    if (a.dims < 3) return std::memcmp(a.data, b.data, size_t(a.w) * a.h * a.elemsize) == 0;
    for (int c = 0; c < a.c; ++c)
        if (std::memcmp(a.channel(c).data, b.channel(c).data,
                        size_t(a.w) * a.h * a.d * a.elemsize)) return false;
    return true;
}
struct Temporary
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ernie-memory-execution-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() { require(std::filesystem::create_directory(path), "Cannot create fixture directory"); }
    ~Temporary() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
std::string inputs(int count)
{
    std::string result;
    for (int i = 0; i < count; ++i)
        result += "Input input" + std::to_string(i) + " 0 1 in" + std::to_string(i) + "\n";
    return result;
}
struct Fixtures
{
    Temporary directory;
    ernie::ComponentFiles gemm;
    ernie::DenoiseModel denoiser;
    Fixtures()
    {
        const auto weights = directory.path / "gemm.bin";
        {
            std::ofstream output(weights, std::ios::binary);
            const std::uint32_t tag = 0;
            const float values[] = {2.f, 0.f, 0.f, 3.f};
            output.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
            output.write(reinterpret_cast<const char*>(values), sizeof(values));
            require(bool(output), "Cannot write GEMM fixture");
        }
        gemm = {"7767517\n11 11\n" + inputs(10) +
            "Gemm projection 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=1 8=2 9=2 10=-1\n", weights.u8string()};
        const auto empty_weights = directory.path / "weightless.bin";
        { std::ofstream output(empty_weights, std::ios::binary); require(bool(output), "Cannot write empty weights"); }
        // A genuine streamed Vulkan denoiser with the production head/constant
        // interfaces. Each 128-channel 1x1 prediction is sample / 8. Timesteps
        // pass through the six required 4096-wide modulation views unchanged.
        denoiser.input_head = {"7767517\n5 11\n" + inputs(3) +
            "Split latent 1 1 in0 out0\n"
            "Split conditioning 1 7 in2 out1 out2 out3 out4 out5 out6 out7\n", empty_weights.u8string()};
        denoiser.output_head = {"7767517\n3 3\n" + inputs(2) +
            "Split output 1 1 in0 out0\n", empty_weights.u8string()};
        denoiser.blocks.assign(3, {"7767517\n11 11\n" + inputs(10) +
            "BinaryOp scale 1 1 in0 out0 0=2 1=1 2=0.5\n", empty_weights.u8string()});
    }
};

#if NCNN_VULKAN
constexpr std::uint64_t mib = 1024ull * 1024;
ernie::HostAvailableReader plenty()
{
    // A deterministic admission input, never an allocation of this size.
    return []() -> std::optional<std::uint64_t> { return 1024 * mib; };
}
ncnn::Option fp32()
{
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_storage = option.use_fp16_packed = option.use_fp16_arithmetic = false;
    option.use_bf16_storage = option.use_bf16_packed = false;
    option.use_packing_layout = false;
    option.num_threads = 1;
    return option;
}
// Single-stream fixture: actually finish the device's work, then report the
// chosen cleanup status. This checks error handling without leaving commands
// in flight or really losing the device, and restores the entry point on exit.
class WaitIdleResult
{
public:
    explicit WaitIdleResult(VkResult result) : original_(ncnn::vkDeviceWaitIdle), result_(result)
    {
        require(!active_, "Nested device wait injection");
        active_ = this;
        ncnn::vkDeviceWaitIdle = intercept;
    }
    ~WaitIdleResult() { ncnn::vkDeviceWaitIdle = original_; active_ = nullptr; }
    WaitIdleResult(const WaitIdleResult&) = delete;
    WaitIdleResult& operator=(const WaitIdleResult&) = delete;
    unsigned calls = 0;
private:
    static VKAPI_ATTR VkResult VKAPI_CALL intercept(VkDevice device)
    {
        auto& self = *active_;
        ++self.calls;
        const auto actual = self.original_(device);
        return actual == VK_SUCCESS ? self.result_ : actual;
    }
    inline static WaitIdleResult* active_ = nullptr;
    PFN_vkDeviceWaitIdle original_;
    VkResult result_;
};
struct Stream
{
    const ncnn::VulkanDevice* device;
    ncnn::VkBlobAllocator blobs;
    ncnn::VkStagingAllocator staging;
    ncnn::Option option = fp32();
    ncnn::VkMat input, constant;
    explicit Stream(const ncnn::VulkanDevice* value) : device(value), blobs(value, 0), staging(value)
    {
        option.blob_vkallocator = option.workspace_vkallocator = &blobs;
        option.staging_vkallocator = &staging;
        ncnn::Mat source(2, 1); source[0] = 1.25f; source[1] = -2.f;
        ncnn::Mat shared(1); shared[0] = 0.f;
        ncnn::VkCompute upload(device);
        upload.record_upload(source, input, option);
        upload.record_upload(shared, constant, option);
        ernie::check_ncnn_memory(upload.submit_and_wait(), "Upload tiny execution fixture");
        require(!input.empty() && !constant.empty(), "Empty execution fixture");
    }
    ncnn::Mat run(const std::vector<ernie::ComponentFiles>& blocks, ernie::BlockSequenceStats& stats,
        const ernie::MemoryExecution& memory, ernie::WeightPlacement* placement = nullptr,
        ernie::WeightSession* cache = nullptr, const ernie::VulkanStageObserver& observer = {})
    {
        const auto result = ernie::run_block_sequence(blocks, input, std::vector<ncnn::VkMat>(9, constant),
            device, option, ernie::WeightPolicy::Stream, stats, observer, placement, cache, &memory);
        ncnn::Mat host;
        ncnn::VkCompute download(device);
        download.record_download(result, host, option);
        ernie::check_ncnn_memory(download.submit_and_wait(), "Download tiny execution fixture");
        require(host.w == 2 && host.h == 1 && host[0] == 10.f && host[1] == -54.f,
                "Streamed 2x2 GEMM result changed");
        return host;
    }
};
bool inspected_host_support(const ncnn::VulkanDevice* device, const ernie::ComponentFiles& files)
{
    ncnn::Net probe;
    probe.opt = fp32();
    probe.opt.use_weights_in_host_memory = true;
    probe.set_vulkan_device(device);
    ernie::load_component(probe, files);
    const auto charge = ernie::dit_host_weight_inspector(device)(probe);
    std::cout << "Tiny host-weight probe charge=" << (charge ? std::to_string(*charge) : "unverified/device-local") << '\n';
    return charge.has_value();
}
void prefetch_contract(const ncnn::VulkanDevice* device, const Fixtures& fixtures)
{
    const bool host_supported = inspected_host_support(device, fixtures.gemm);
    Stream stream(device);
    const std::vector<ernie::ComponentFiles> blocks(3, fixtures.gemm);
    ernie::MemoryExecution memory;
    memory.ram_reserve_bytes = 0;
    memory.available = plenty();
    ernie::BlockSequenceStats baseline_stats;
    const auto baseline = stream.run(blocks, baseline_stats, memory);
    require(baseline_stats.prefetch_started == 0 && baseline_stats.peak_loaded_nets == 1,
            "Disabled prefetch did not retain single-block streaming");

    memory.prefetch_bytes = 96 * mib;
    const auto owner_thread = std::this_thread::get_id();
    int traced_prefetch = 0;
    ernie::WeightPlacement placement(ernie::WeightMemory::Auto, 0, {},
        [&](const ernie::ComponentFiles&, const ernie::WeightPlacementDecision& decision) {
            require(std::this_thread::get_id() == owner_thread, "Background loader mutated placement observer");
            if (std::string(decision.reason) == "prefetch") ++traced_prefetch;
        });
    memory.available = [&]() -> std::optional<std::uint64_t> {
        require(std::this_thread::get_id() == owner_thread, "Background loader read mutable admission state");
        return 1024 * mib;
    };
    ernie::BlockSequenceStats prepared;
    require(exact(stream.run(blocks, prepared, memory, &placement), baseline), "Prefetch changed exact output");
    require(prepared.prefetch_started == 2 && traced_prefetch == 2 && prepared.peak_loaded_nets == 2 &&
            prepared.compute_submissions == 3 && prepared.prefetch_peak_charged_bytes <= memory.prefetch_bytes,
            "One-block prefetch bounds/accounting differ");
    require(prepared.prefetch_used == (host_supported ? 2u : 0u), "Unverified or device-local weights entered prefetch");
    require(prepared.prefetch_overlap_seconds >= 0 && std::isfinite(prepared.prefetch_overlap_seconds),
            "Invalid preparation-overlap observation");
    if (host_supported) require(prepared.load_seconds.size() == 3, "Prefetch loaded a consumed block twice");

    memory.prefetch_bytes = 1;
    ernie::BlockSequenceStats small;
    stream.run(blocks, small, memory);
    require(small.prefetch_started == 0 && small.prefetch_skipped == 2, "Prefetch exceeded a one-byte budget");
    memory.prefetch_bytes = 96 * mib;
    memory.available = {};
    ernie::BlockSequenceStats unknown;
    stream.run(blocks, unknown, memory);
    require(unknown.prefetch_started == 0 && unknown.prefetch_skipped == 2, "Unknown RAM headroom admitted prefetch");
    memory.available = plenty();
    ernie::WeightPlacement forced_device(ernie::WeightMemory::Device, 0);
    ernie::BlockSequenceStats forced;
    stream.run(blocks, forced, memory, &forced_device);
    require(forced.prefetch_started == 0 && forced.prefetch_skipped == 2,
            "Prefetch overrode explicit device weight placement");

    int reads = 0;
    memory.ram_reserve_bytes = mib;
    memory.available = [&]() -> std::optional<std::uint64_t> { return ++reads == 1 ? 1024 * mib : 0; };
    ernie::BlockSequenceStats pressure;
    stream.run(blocks, pressure, memory);
    require(pressure.prefetch_started == 1 && pressure.prefetch_used == 0 && pressure.prefetch_skipped == 2,
            "Prepared block ignored new RAM pressure at consumption");
    memory.ram_reserve_bytes = 0;
    memory.available = plenty();

    ernie::WeightSession cache({256 * mib, 0}, plenty(), ernie::dit_host_weight_inspector(device));
    ernie::BlockSequenceStats first, second;
    stream.run(blocks, first, memory, nullptr, &cache);
    stream.run(blocks, second, memory, nullptr, &cache);
    if (host_supported)
    {
        require(cache.stats().hits == 3 && cache.stats().loads == 3 && second.prefetch_started == 0 &&
                std::all_of(second.load_seconds.begin(), second.load_seconds.end(), [](double seconds) { return seconds == 0; }),
                "Cached blocks were redundantly prepared in the background");
    }
    else require(cache.stats().hits == 0 && second.prefetch_used == 0, "Device-local weights entered RAM reuse");
    cache.cancel();
    require(cache.stats().live_bytes == 0, "Cancelling prefetch/cache retained idle weights");

    auto broken = blocks;
    broken[1].param_text = "7767516\n0 0\n";
    int completed = 0;
    ernie::BlockSequenceStats failure;
    const auto message = expect_failure<std::runtime_error>([&] {
        stream.run(broken, failure, memory, nullptr, nullptr,
            [&](const std::string&, const ncnn::VkMat&) { ++completed; });
    });
    require(failure.prefetch_started == 1 && failure.prefetch_used == 0 && failure.prefetch_skipped == 1 &&
            completed == 1 && message.find("graph failed") != std::string::npos,
            "Background graph error was lost or failed before real first-block computation");
    // Abandon a sequence while its next Net is being prepared. Unwinding must
    // join preparation before caller state is destroyed; a fresh request on
    // the same device then exercises the resulting lifetimes under validation.
    {
        Stream abandoned(device);
        ernie::BlockSequenceStats interrupted;
        const auto error = expect_failure<std::logic_error>([&] {
            abandoned.run(blocks, interrupted, memory, nullptr, nullptr,
                [&](const std::string&, const ncnn::VkMat&) {
                    require(interrupted.prefetch_started == 1, "No outstanding prefetch at cancellation");
                    throw std::logic_error("cancel during preparation");
                });
        });
        require(error == "cancel during preparation", "Unwind replaced the primary callback exception");
    }
    ernie::BlockSequenceStats after_cancel;
    require(exact(stream.run(blocks, after_cancel, memory), baseline), "Cancelled prefetch poisoned subsequent execution");
    std::cout << "Prefetch: exact GEMM, bounded admission, current headroom, inspected residency, cache and unwind passed\n";
}

ncnn::Mat initial_latent()
{
    ncnn::Mat result(1, 1, 128);
    for (int c = 0; c < result.c; ++c) result.channel(c)[0] = float(c % 13 - 6) * 0.125f + 0.03125f;
    return result;
}
ernie::VulkanDenoiseSettings settings()
{
    ernie::VulkanDenoiseSettings result;
    result.memory = ernie::ActivationMemory::Device;
    result.weights = ernie::WeightMemory::Device;
    result.gpu_reserve_bytes = result.ram_reserve_bytes = 0;
    result.spill_bytes = 8 * mib;
    result.available = plenty();
    result.download_predictions = true;
    return result;
}
void recovery_contract(const ncnn::VulkanDevice* device, const Fixtures& fixtures)
{
    const auto option = fp32();
    const auto initial = initial_latent();
    ncnn::Mat constant(1); constant[0] = 0.f;
    const std::vector<ncnn::Mat> constants(4, constant);
    constexpr int steps = 4;
    const auto schedule = ernie::FlowSchedule::turbo(steps);
    auto run = [&](const ernie::VulkanDenoiseSettings& configuration, ernie::VulkanDenoiseStats& stats,
                   const ernie::VulkanDenoiseObservers& observer = {}, int start = 0,
                   const ncnn::Mat& checkpoint = ncnn::Mat()) {
        return ernie::denoise_with_recovery(fixtures.denoiser, checkpoint.empty() ? initial : checkpoint,
            constants, steps, start, device, option, configuration, stats, observer);
    };
    auto base = settings();
    ernie::VulkanDenoiseStats baseline_stats;
    ernie::VulkanDenoiseObservers baseline_observer;
    std::vector<ncnn::Mat> checkpoints;
    baseline_observer.step = [&](size_t index, const ncnn::Mat& prediction, const ncnn::Mat& value,
                                 const ernie::DenoiseStepStats& step) {
        require(index == checkpoints.size() && step.complete && step.timestep == schedule.timesteps[index] &&
                step.delta == schedule.delta(index) && !prediction.empty(), "Baseline schedule or checkpoint differs");
        checkpoints.push_back(value.clone());
    };
    const auto baseline = run(base, baseline_stats, baseline_observer);
    require(checkpoints.size() == steps && baseline_stats.retries == 0 && baseline_stats.query_rows == 128 &&
            !exact(baseline, initial), "Tiny model did not execute real nonzero denoising");
    for (int c = 0; c < initial.c; ++c)
    {
        float expected = initial.channel(c)[0];
        for (int i = 0; i < steps; ++i) expected += schedule.delta(i) * (expected * 0.125f);
        require(std::abs(baseline.channel(c)[0] - expected) <= 3e-7f, "Tiny denoiser differs from independent Euler calculation");
    }

    std::vector<std::pair<unsigned, int>> attempted;
    std::vector<size_t> committed;
    std::vector<std::tuple<unsigned, int, int>> retries;
    auto injected = base;
    injected.memory = ernie::ActivationMemory::Auto;
    injected.weights = ernie::WeightMemory::Auto;
    injected.before_step = [&](unsigned attempt, int step) {
        attempted.emplace_back(attempt, step);
        if (attempt == 0 && step == 2) throw ernie::GpuAllocationError("controlled allocation failure after step 1");
    };
    ernie::VulkanDenoiseObservers observer;
    observer.retry = [&](unsigned attempt, int step, int rows, const std::string&) { retries.emplace_back(attempt, step, rows); };
    observer.step = [&](size_t index, const ncnn::Mat&, const ncnn::Mat& value, const ernie::DenoiseStepStats& step) {
        committed.push_back(index);
        require(step.timestep == schedule.timesteps[index] && step.delta == schedule.delta(index) &&
                exact(value, checkpoints.at(index)), "Recovery changed an absolute timestep or completed latent");
    };
    ernie::VulkanDenoiseStats recovered;
    require(exact(run(injected, recovered, observer), baseline), "Allocation recovery changed final tensor bits");
    require(attempted == std::vector<std::pair<unsigned, int>>{{0, 0}, {0, 1}, {0, 2}, {1, 2}, {1, 3}} &&
            committed == std::vector<size_t>{0, 1, 2, 3} &&
            retries == std::vector<std::tuple<unsigned, int, int>>{{1, 2, 64}} &&
            recovered.retries == 1 && recovered.query_rows == 64 && recovered.memory.host_allocations > 0,
            "Recovery repeated a committed step, failed to shrink chunks or failed to reconstruct in RAM");

    attempted.clear(); committed.clear(); retries.clear();
    injected.before_step = [&](unsigned attempt, int step) {
        attempted.emplace_back(attempt, step);
        if (!attempt && step == 3) throw std::bad_alloc();
    };
    ernie::VulkanDenoiseStats resumed;
    require(exact(run(injected, resumed, observer, 2, checkpoints[1]), baseline), "Nonzero start-step recovery changed output");
    require(attempted == std::vector<std::pair<unsigned, int>>{{0, 2}, {0, 3}, {1, 3}} &&
            committed == std::vector<size_t>{2, 3} && resumed.retries == 1,
            "Nonzero start-step recovery renumbered or replayed the schedule");

    for (unsigned limit : {0u, 3u})
    {
        auto bounded = base;
        bounded.retries = limit;
        unsigned calls = 0;
        bounded.before_step = [&](unsigned attempt, int step) {
            require(attempt == calls++ && step == 0, "Failed restart changed checkpoint or attempt index");
            throw ernie::GpuAllocationError("persistent controlled failure");
        };
        ernie::VulkanDenoiseStats exhausted;
        const auto error = expect_failure<ernie::GpuAllocationError>([&] { run(bounded, exhausted); });
        require(calls == limit + 1 && exhausted.retries == limit && exhausted.query_rows == (128 >> limit) &&
                error.find("exhausted") != std::string::npos, "Recovery exceeded its retry/chunk bounds");
    }

    for (bool device_lost : {false, true})
    {
        auto fatal = base;
        unsigned calls = 0;
        fatal.before_step = [&](unsigned, int) {
            ++calls;
            if (device_lost) ernie::check_vulkan_memory(VK_ERROR_DEVICE_LOST, "injected device loss");
            else ernie::check_ncnn_memory(-1, "injected generic ncnn failure");
        };
        ernie::VulkanDenoiseStats stats;
        bool ordinary = false;
        try { run(fatal, stats); }
        catch (const ernie::GpuAllocationError&) { throw std::runtime_error("Generic/device-lost error was classified as OOM"); }
        catch (const std::runtime_error&) { ordinary = true; }
        require(ordinary && calls == 1 && stats.retries == 0, "Fatal computation error was retried");
    }

    for (const auto status : {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_DEVICE_LOST})
    {
        auto failed_cleanup = base;
        int attempted_steps = 0, retry_calls = 0;
        failed_cleanup.before_step = [&](unsigned, int) {
            ++attempted_steps;
            throw ernie::GpuAllocationError("original allocation failure before cleanup");
        };
        ernie::VulkanDenoiseObservers callbacks;
        callbacks.retry = [&](unsigned, int, int, const std::string&) { ++retry_calls; };
        ernie::VulkanDenoiseStats stats;
        WaitIdleResult wait(status);
        std::string message;
        try { run(failed_cleanup, stats, callbacks); }
        catch (const ernie::GpuAllocationError&) { throw std::logic_error("A failed cleanup wait remained recoverable"); }
        catch (const std::runtime_error& error) { message = error.what(); }
        require(message.find("cannot safely retry") != std::string::npos &&
                message.find(std::to_string(status)) != std::string::npos && wait.calls == 1 &&
                attempted_steps == 1 && retry_calls == 0 && stats.retries == 0,
                "Cleanup wait failure lost its status or retried an unsynchronized attempt");
    }
    {
        auto original_failure = base;
        original_failure.before_step = [](unsigned, int) { throw std::logic_error("original fatal step failure"); };
        ernie::VulkanDenoiseStats stats;
        WaitIdleResult wait(VK_SUCCESS);
        const auto message = expect_failure<std::logic_error>([&] { run(original_failure, stats); });
        require(message == "original fatal step failure" && wait.calls == 1 && stats.retries == 0,
                "Successful cleanup replaced the original exception");
    }

    // Callback exceptions describe user code or reporting, even when their
    // type happens to be bad_alloc. They must never trigger another inference.
    for (int callback = 0; callback < 5; ++callback)
    {
        auto configuration = base;
        int callback_calls = 0, retry_calls = 0;
        ernie::VulkanDenoiseObservers callbacks;
        const auto fail = [&] { ++callback_calls; throw std::bad_alloc(); };
        callbacks.retry = [&](unsigned, int, int, const std::string&) { ++retry_calls; };
        if (callback == 0) callbacks.transfer = [&](bool upload, double) { if (upload) fail(); };
        if (callback == 1) callbacks.transfer = [&](bool upload, double) { if (!upload) fail(); };
        if (callback == 2) callbacks.placement = [&](const ernie::ComponentFiles&, const ernie::WeightPlacementDecision&) { fail(); };
        if (callback == 3) callbacks.step = [&](size_t, const ncnn::Mat&, const ncnn::Mat&, const ernie::DenoiseStepStats&) { fail(); };
        if (callback == 4)
        {
            configuration.before_step = [](unsigned, int) { throw std::runtime_error("original computation failure"); };
            callbacks.failed_step = [&](size_t, const ernie::DenoiseStepStats&) { fail(); };
        }
        ernie::VulkanDenoiseStats stats;
        expect_failure<std::bad_alloc>([&] { run(configuration, stats, callbacks); });
        require(callback_calls == 1 && retry_calls == 0 && stats.retries == 0,
                "An observer failure triggered retry or duplicate notification");
    }

    for (bool unknown : {false, true})
    {
        auto refused = base;
        refused.memory = ernie::ActivationMemory::Host;
        refused.spill_bytes = unknown ? 8 * mib : 1;
        if (unknown) refused.available = {};
        refused.retries = 2;
        unsigned steps_attempted = 0;
        refused.before_step = [&](unsigned, int) { ++steps_attempted; };
        ernie::VulkanDenoiseStats stats;
        expect_failure<ernie::GpuAllocationError>([&] { run(refused, stats); });
        require(stats.retries == 2 && stats.memory.allocation_failures >= 3 && steps_attempted == 0,
                "Host budget refusal was unbounded or dispatched incomplete inputs");
    }
    ernie::VulkanDenoiseStats no_work;
    require(exact(run(base, no_work, {}, steps, baseline), baseline) && no_work.retries == 0,
            "A complete checkpoint performed another model step");
    std::cout << "Recovery: real Euler steps, exact resumed output, absolute schedule, bounded failures and fatal callbacks passed\n";
}
#endif
} // namespace

int main()
{
    try
    {
#if NCNN_VULKAN
        struct Context { ~Context() { ncnn::destroy_gpu_instance(); } } context;
        if (ncnn::create_gpu_instance() || ncnn::get_gpu_count() == 0) return 77;
        const auto* device = ncnn::get_gpu_device();
        Fixtures fixtures;
        prefetch_contract(device, fixtures);
        recovery_contract(device, fixtures);
        return 0;
#else
        return 77;
#endif
    }
    catch (const std::exception& error)
    {
        std::cerr << "Memory execution contract: " << error.what() << '\n';
        return 1;
    }
}

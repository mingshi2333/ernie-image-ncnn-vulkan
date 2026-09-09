// SPDX-License-Identifier: MIT
#include "vulkan_memory.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#if NCNN_VULKAN
#include "command.h"
#include "net.h"
#endif

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
template<class F> void allocation_fails(F&& action)
{
    try { action(); }
    catch (const ernie::GpuAllocationError&) { return; }
    throw std::runtime_error("Expected a recoverable allocation failure");
}
template<class F> void fatal_fails(F&& action)
{
    try { action(); }
    catch (const ernie::GpuAllocationError&) { throw std::runtime_error("Non-memory error was marked recoverable"); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error("Expected a fatal execution failure");
}

void status_contract()
{
    using namespace ernie;
    require(parse_activation_memory("auto") == ActivationMemory::Auto &&
            parse_activation_memory("device") == ActivationMemory::Device &&
            parse_activation_memory("host") == ActivationMemory::Host, "Activation mode parsing differs");
    bool invalid = false;
    try { parse_activation_memory("cpu"); } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "Activation mode silently accepted a CPU fallback");
    check_ncnn_memory(0, "Synthetic ncnn success");
    check_vulkan_memory(0, "Synthetic Vulkan success");
    allocation_fails([] { check_ncnn_memory(-100, "ncnn allocation"); });
    fatal_fails([] { check_ncnn_memory(-1, "ncnn extraction"); });
    fatal_fails([] { check_ncnn_memory(-2, "ncnn graph"); });
    allocation_fails([] { check_vulkan_memory(-1, "Vulkan host allocation"); });
    allocation_fails([] { check_vulkan_memory(-2, "Vulkan device allocation"); });
    fatal_fails([] { check_vulkan_memory(-4, "Vulkan device lost"); });
    fatal_fails([] { check_vulkan_memory(-3, "Vulkan initialization"); });
    fatal_fails([] { check_vulkan_memory(2, "Vulkan wait timeout"); });
    std::cout << "Activation mode and ncnn/Vulkan error classification contracts passed\n";
}

#if NCNN_VULKAN
constexpr std::uint64_t mib = 1024ull * 1024;
ernie::HostAvailableReader plenty()
{
    return []() -> std::optional<std::uint64_t> { return 128 * mib; };
}
ernie::AdaptiveVkAllocator::BudgetReader pressure()
{
    // A controlled decision boundary, not an attempt to exhaust the GPU.
    return []() -> std::optional<ernie::DeviceMemoryBudget> { return ernie::DeviceMemoryBudget{1024, 1024}; };
}

// This executable owns one Vulkan stream. Fail only the next matching ncnn
// allocation call; do not fill real VRAM or change the driver/device state.
class AllocationFailure
{
public:
    AllocationFailure(std::uint32_t type, VkResult error)
        : original_(ncnn::vkAllocateMemory), type_(type), error_(error)
    {
        require(!active_, "Nested Vulkan allocation injection");
        active_ = this;
        ncnn::vkAllocateMemory = intercept;
    }
    ~AllocationFailure() { ncnn::vkAllocateMemory = original_; active_ = nullptr; }
    AllocationFailure(const AllocationFailure&) = delete;
    AllocationFailure& operator=(const AllocationFailure&) = delete;
    bool fired = false;
private:
    static VKAPI_ATTR VkResult VKAPI_CALL intercept(VkDevice device, const VkMemoryAllocateInfo* info,
        const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory)
    {
        auto& self = *active_;
        if (!self.fired && info->memoryTypeIndex == self.type_)
        {
            self.fired = true;
            *memory = VK_NULL_HANDLE;
            return self.error_;
        }
        return self.original_(device, info, callbacks, memory);
    }
    inline static AllocationFailure* active_ = nullptr;
    PFN_vkAllocateMemory original_;
    std::uint32_t type_;
    VkResult error_;
};

void failure_contract(const ncnn::VulkanDevice* device)
{
    using namespace ernie;
    auto available = []() -> std::optional<DeviceMemoryBudget> { return DeviceMemoryBudget{UINT64_MAX, 0}; };
    AdaptiveVkAllocator automatic(device, ActivationMemory::Auto, 0, mib, 0, plenty(), available);
    {
        AllocationFailure fault(automatic.buffer_memory_type_index, VK_ERROR_OUT_OF_DEVICE_MEMORY);
        const auto reader = compute_memory_budget_reader(device);
        (void)reader();
        require(!fault.fired, "Compute heap query allocated device memory during recovery");
        auto* ptr = automatic.fastMalloc(64);
        require(fault.fired && automatic.stats().allocation_failures == 1 &&
            automatic.stats().host_allocations == 1 && automatic.stats().fallbacks == 1 &&
            automatic.stats().device_allocations == 0, "Actual Vulkan OOM did not fall back into bounded host storage");
        automatic.fastFree(ptr);
        automatic.reclaim_completed();
    }
    auto* resumed = automatic.fastMalloc(64);
    automatic.fastFree(resumed);
    automatic.clear();
    require(automatic.stats().device_allocations == 1 && automatic.stats().host_live_bytes == 0,
            "Device allocator was poisoned by a failed allocation");

    AdaptiveVkAllocator strict(device, ActivationMemory::Device, 0, mib, 0, plenty(), available);
    {
        AllocationFailure fault(strict.buffer_memory_type_index, VK_ERROR_OUT_OF_DEVICE_MEMORY);
        allocation_fails([&] { strict.fastMalloc(64); });
        require(fault.fired && strict.stats().allocation_failures == 1 && strict.stats().fallbacks == 0 &&
            strict.stats().host_allocations == 0, "Device-only allocation silently fell back");
    }
    // These injected results do not actually lose the GPU. They verify that
    // the ncnn failure wrapper and adaptive allocator preserve fatal errors.
    for (const auto code : {VK_ERROR_DEVICE_LOST, VK_ERROR_TOO_MANY_OBJECTS})
    {
        AllocationFailure fault(automatic.buffer_memory_type_index, code);
        fatal_fails([&] { automatic.fastMalloc(64); });
        require(fault.fired && automatic.stats().fallbacks == 1 && automatic.stats().host_allocations == 1,
                "Non-OOM Vulkan error triggered a host fallback");
    }
    AdaptiveVkAllocator host(device, ActivationMemory::Host, 0, mib, 0, plenty());
    auto* probe = host.fastMalloc(64);
    const auto host_type = probe->memory_type_index;
    host.fastFree(probe);
    host.clear();
    {
        AllocationFailure fault(host_type, VK_ERROR_OUT_OF_HOST_MEMORY);
        allocation_fails([&] { host.fastMalloc(64); });
        require(fault.fired && host.stats().allocation_failures == 1 && host.stats().host_live_bytes == 0,
                "Failed direct host allocation leaked its budget or lost its error class");
    }
    std::cout << "Injected device allocation OOM recovers; explicit device mode and non-OOM errors retain their boundaries\n";
}

std::uint64_t budget_contract(const ncnn::VulkanDevice* device)
{
    using namespace ernie;
    AdaptiveVkAllocator allocator(device, ActivationMemory::Host, 0, mib, 0, plenty());
    const auto* properties = &device->info.physicalDeviceMemoryProperties();
    auto* ptr = allocator.fastMalloc(64);
    const auto flags = properties->memoryTypes[ptr->memory_type_index].propertyFlags;
    require((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        "Spilled activation has no coherent host-visible storage");
    // The real buffer's compatible memoryTypeBits determine whether a
    // non-device-local heap exists; report actual residency on UMA too.
    VkMemoryRequirements requirements{};
    ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), ptr->buffer, &requirements);
    bool nonlocal_available = false;
    for (std::uint32_t index = 0; index < properties->memoryTypeCount; ++index)
    {
        const auto candidate = properties->memoryTypes[index].propertyFlags;
        if ((requirements.memoryTypeBits & (std::uint32_t(1) << index)) &&
            (candidate & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            !(candidate & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) nonlocal_available = true;
    }
    require(!nonlocal_available || !(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            "Host activation ignored compatible non-device-local memory");
    const auto charged = allocator.stats().host_live_bytes;
    require(charged == requirements.size && charged >= ptr->capacity &&
        allocator.stats().host_allocations == 1 && allocator.stats().device_allocations == 0,
        "Host accounting used tensor bytes instead of actual Vulkan allocation");
    require(allocator.flush(ptr) == 0 && allocator.invalidate(ptr) == 0,
            "Coherent host memory flush/invalidate contract differs");
    allocator.clear();
    require(allocator.stats().host_live_bytes == charged, "Clear destroyed an active activation");
    allocator.fastFree(ptr);
    require(allocator.stats().host_live_bytes == charged, "Release destroyed a potentially recorded buffer");
    reclaim_completed(&allocator);
    require(allocator.stats().host_live_bytes == 0 && allocator.stats().host_peak_bytes == charged,
            "Completed host memory was not reclaimed accurately");

    AdaptiveVkAllocator bounded(device, ActivationMemory::Host, 0, charged, 10,
        [charged]() -> std::optional<std::uint64_t> { return charged + 10; });
    auto* exact = bounded.fastMalloc(64);
    allocation_fails([&] { bounded.fastMalloc(64); });
    bounded.fastFree(exact);
    allocation_fails([&] { bounded.fastMalloc(64); });
    bounded.reclaim_completed();
    exact = bounded.fastMalloc(64);
    bounded.fastFree(exact);
    bounded.clear();
    require(bounded.stats().host_live_bytes == 0 && bounded.stats().host_peak_bytes == charged &&
        bounded.stats().allocation_failures == 2, "Bounded RAM admission/release accounting differs");

    AdaptiveVkAllocator disabled(device, ActivationMemory::Host, 0, 0, 0, plenty());
    allocation_fails([&] { disabled.fastMalloc(64); });
    AdaptiveVkAllocator missing(device, ActivationMemory::Host, 0, mib, 0,
        []() -> std::optional<std::uint64_t> { return std::nullopt; });
    allocation_fails([&] { missing.fastMalloc(64); });
    AdaptiveVkAllocator insufficient(device, ActivationMemory::Host, 0, mib, 10,
        [charged]() -> std::optional<std::uint64_t> { return charged + 9; });
    allocation_fails([&] { insufficient.fastMalloc(64); });
    AdaptiveVkAllocator overflow(device, ActivationMemory::Host, 0, UINT64_MAX, UINT64_MAX,
        []() -> std::optional<std::uint64_t> { return UINT64_MAX; });
    allocation_fails([&] { overflow.fastMalloc(64); });
    require(disabled.stats().host_live_bytes == 0 && missing.stats().host_live_bytes == 0 &&
        insufficient.stats().host_live_bytes == 0 && overflow.stats().host_live_bytes == 0,
        "Refused allocations consumed the RAM budget");
    std::cout << "Actual host memory type=" << ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ?
        "host-visible device-local (unified memory)" : "non-device-local")
        << " Vulkan charge=" << charged << " bytes; release and refusal contracts passed\n";
    return charged;
}

void reuse_contract(const ncnn::VulkanDevice* device)
{
    using namespace ernie;
    bool pressured = false;
    unsigned reads = 0;
    AdaptiveVkAllocator allocator(device, ActivationMemory::Auto, 0, 4 * mib, 0, plenty(),
        [&]() -> std::optional<DeviceMemoryBudget> {
            ++reads;
            return DeviceMemoryBudget{UINT64_MAX, pressured ? UINT64_MAX : 0};
        });
    constexpr size_t unit = 64 * 1024;
    auto* whole = allocator.fastMalloc(4 * unit);
    const auto memory = whole->memory;
    const auto buffer = whole->buffer;
    require(whole->offset == 0 && whole->capacity == 4 * unit && reads == 1,
            "Initial device pool backing differs");
    allocator.fastFree(whole);
    pressured = true;

    // Existing VkBuffer + VkDeviceMemory handles and offsets identify actual
    // suballocations of the original backing, not newly allocated storage.
    // The reviewed ncnn pool returns these regions before vkAllocateMemory;
    // this test does not replace any Vulkan function or rely on noisy global
    // heap telemetry to infer whether reuse happened.
    ncnn::VkBufferMemory* pieces[4];
    for (size_t index = 0; index < 4; ++index)
    {
        pieces[index] = allocator.fastMalloc(unit);
        require(pieces[index]->buffer == buffer && pieces[index]->memory == memory &&
                pieces[index]->offset == index * unit && pieces[index]->capacity == unit,
                "Pressure ignored an existing contiguous device region");
    }
    require(reads == 1 && allocator.stats().host_allocations == 0,
            "Existing device regions were charged as new backing memory");
    auto* full = allocator.fastMalloc(unit);
    require(reads == 2 && allocator.stats().host_allocations == 1 &&
            allocator.stats().fallbacks == 1 && full->memory != memory,
            "An entirely active device backing was reused or grew past the budget");

    allocator.fastFree(pieces[0]);
    allocator.fastFree(pieces[2]);
    auto* fragmented = allocator.fastMalloc(2 * unit);
    require(reads == 3 && allocator.stats().host_allocations == 2 && fragmented->memory != memory,
            "Separated free regions were incorrectly treated as contiguous storage");
    allocator.fastFree(pieces[1]);
    auto* merged = allocator.fastMalloc(3 * unit);
    require(reads == 3 && merged->buffer == buffer && merged->memory == memory && merged->offset == 0 &&
            merged->capacity == 3 * unit && pieces[3]->offset == 3 * unit,
            "Adjacent released regions were not reused or overlapped a live range");

    allocator.clear(); // Active pieces keep the pool and its range knowledge.
    allocator.fastFree(merged);
    auto* after_active_clear = allocator.fastMalloc(3 * unit);
    require(reads == 3 && after_active_clear->memory == memory && after_active_clear->buffer == buffer &&
            after_active_clear->offset == 0,
            "Clear discarded reusable range knowledge while a device tensor was active");
    allocator.fastFree(after_active_clear);
    allocator.fastFree(pieces[3]);
    allocator.fastFree(full);
    allocator.fastFree(fragmented);
    allocator.clear(); // No live device tensors: backing and knowledge both go.
    require(allocator.stats().host_live_bytes == 0, "Reuse fixture leaked host spill storage");
    auto* cleared = allocator.fastMalloc(unit);
    require(reads == 4 && allocator.stats().host_allocations == 3,
            "A cleared backing was still treated as available device storage");
    allocator.fastFree(cleared);
    allocator.clear();

    // Free bytes from separate buffers cannot satisfy one larger request.
    pressured = false;
    auto* first = allocator.fastMalloc(unit);
    auto* second = allocator.fastMalloc(unit);
    require(first->memory != second->memory, "Two active device backings alias");
    allocator.fastFree(first);
    allocator.fastFree(second);
    pressured = true;
    auto* separate = allocator.fastMalloc(2 * unit);
    require(reads == 7 && allocator.stats().host_allocations == 4 &&
            allocator.stats().allocation_failures == 0,
            "Free space from separate device buffers bypassed the backing budget");
    allocator.fastFree(separate);
    allocator.clear();
    std::cout << "Device reuse under pressure preserves backing handles, contiguous ranges, active tensors and clear boundaries\n";
}

void compute_contract(const ncnn::VulkanDevice* device, ernie::ActivationMemory mode,
                      ernie::AdaptiveVkAllocator::BudgetReader budget, bool expect_host)
{
    using namespace ernie;
    AdaptiveVkAllocator allocator(device, mode, 0, 8 * mib, 0, plenty(), std::move(budget));
    ncnn::VkStagingAllocator staging(device);
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_fp16_storage = option.use_fp16_packed = option.use_fp16_arithmetic = false;
    option.use_bf16_storage = option.use_bf16_packed = false;
    option.use_packing_layout = false;
    option.num_threads = 1;
    option.blob_vkallocator = option.workspace_vkallocator = &allocator;
    option.staging_vkallocator = &staging;
    ncnn::Net net;
    net.set_vulkan_device(device);
    net.opt = option;
    // ncnn lightmode requires an explicit Split when a value feeds two
    // branches; it also prevents scalar in-place execution mutating residual.
    const char* graph = "7767517\n5 6\nInput input 0 1 in\n"
        "Split branches 1 2 in compute residual\n"
        "BinaryOp double 1 1 compute doubled 0=2 1=1 2=2.0\n"
        "BinaryOp offset 1 1 doubled shifted 0=0 1=1 2=1.0\n"
        "BinaryOp product 2 1 shifted residual out 0=2\n";
    check_ncnn_memory(net.load_param_mem(graph), "Load activation graph");
    const unsigned char empty_weights[4] = {0, 0, 0, 0};
    require(net.load_model(empty_weights) == 0, "Load activation graph model");

    ncnn::Mat input(256), result;
    for (int i = 0; i < input.w; ++i) input[i] = (i - 128) * .125f;
    {
        ncnn::VkMat gpu_input, copied, output;
        {
            ncnn::VkCompute command(device);
            command.record_clone(input, gpu_input, option);
            // This temporary must remain backed until command completion,
            // although its last VkMat reference is dropped before submission.
            ncnn::VkMat temporary;
            command.record_clone(gpu_input, temporary, option);
            command.record_clone(temporary, copied, option);
            temporary.release();
            auto extractor = net.create_extractor();
            check_ncnn_memory(extractor.input("in", copied), "Feed activation fixture");
            check_ncnn_memory(extractor.extract("out", output, command), "Execute activation fixture");
            require(!output.empty(), "GPU activation output is empty");
            command.record_download(output, result, option);
            check_ncnn_memory(command.submit_and_wait(), "Submit activation fixture");
        }
        allocator.reclaim_completed();
        if (expect_host)
            require(allocator.stats().host_live_bytes > 0, "Reclaim destroyed active GPU tensors");
        require(result.w == input.w, "Activation fixture output shape differs");
        for (int i = 0; i < input.w; ++i)
            require(result[i] == (input[i] * 2.f + 1.f) * input[i],
                "GPU kernels using activation RAM changed the exact result");
    }
    allocator.clear();
    const auto stats = allocator.stats();
    require(stats.host_live_bytes == 0 && stats.host_peak_bytes <= 8 * mib && stats.allocation_failures == 0,
        "GPU execution leaked/exceeded bounded activation RAM");
    if (expect_host)
        require(stats.host_allocations > 0 && stats.device_allocations == 0 &&
            stats.host_device_local_allocations + stats.host_non_device_local_allocations == stats.host_allocations,
            "Host/auto pressure fixture did not compute on host-visible buffers");
    else require(stats.device_allocations > 0 && stats.host_allocations == 0,
            "Device mode unexpectedly spilled activation buffers");
    require(mode != ActivationMemory::Auto || (expect_host ? stats.fallbacks > 0 : stats.fallbacks == 0),
        "Automatic activation placement counters differ");
    std::cout << "GPU compute/copy exact; mode=" << static_cast<int>(mode)
        << " device=" << stats.device_allocations << " host=" << stats.host_allocations
        << " fallbacks=" << stats.fallbacks << " host peak=" << stats.host_peak_bytes << '\n';
}

int vulkan_contract()
{
    struct Instance { ~Instance() { ncnn::destroy_gpu_instance(); } } instance;
    if (ncnn::create_gpu_instance() || ncnn::get_gpu_count() == 0) return 77;
    const auto* device = ncnn::get_gpu_device();
    failure_contract(device);
    budget_contract(device);
    reuse_contract(device);
    compute_contract(device, ernie::ActivationMemory::Host, {}, true);
    compute_contract(device, ernie::ActivationMemory::Auto, pressure(), true);
    compute_contract(device, ernie::ActivationMemory::Device, pressure(), false);
    compute_contract(device, ernie::ActivationMemory::Auto,
        []() -> std::optional<ernie::DeviceMemoryBudget> { return std::nullopt; }, false);

    // Re-query pressure on every allocation; the next tensor can return to
    // device storage while the previous spilled tensor stays valid.
    int calls = 0;
    ernie::AdaptiveVkAllocator changing(device, ernie::ActivationMemory::Auto, 0, mib, 0, plenty(),
        [&]() -> std::optional<ernie::DeviceMemoryBudget> {
            return ernie::DeviceMemoryBudget{UINT64_MAX, ++calls == 1 ? UINT64_MAX : 0};
        });
    auto* host = changing.fastMalloc(64);
    auto* gpu = changing.fastMalloc(64);
    require(changing.stats().host_allocations == 1 && changing.stats().device_allocations == 1 && calls == 2,
            "Activation policy did not observe changing pressure");
    changing.fastFree(host);
    changing.fastFree(gpu);
    changing.clear();
    require(changing.stats().host_live_bytes == 0, "Changing placement leaked host memory");
    // Image-storage production is disabled, but the required virtual overloads
    // must delegate allocation and release to the same owner.
    {
        ncnn::VkImageMat image(4, 4, 1, size_t(4), 1, &changing);
        require(!image.empty(), "Delegated image allocation is empty");
    }
    changing.clear();
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
        status_contract();
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

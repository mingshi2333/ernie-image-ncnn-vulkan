// SPDX-License-Identifier: MIT
// Deterministic Vulkan error injection; this does not exhaust physical memory.
#include "allocator.h"
#include "command.h"
#include "gpu.h"
#include "option.h"
#include <atomic>
#include <exception>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>

namespace {
VkResult injected = VK_ERROR_OUT_OF_DEVICE_MEMORY;
unsigned failures_left = 0;
unsigned injections = 0;
PFN_vkAllocateMemory allocate_original;
PFN_vkCreateBuffer create_buffer_original;
PFN_vkMapMemory map_original;
PFN_vkCreateCommandPool create_pool_original;
PFN_vkCreateFence create_fence_original;
PFN_vkQueueSubmit submit_original;
PFN_vkWaitForFences wait_original;
PFN_vkDestroyBuffer destroy_buffer_original;
PFN_vkFreeMemory free_memory_original;
unsigned buffers_created = 0, buffers_destroyed = 0;
unsigned memory_created = 0, memory_destroyed = 0;

bool fail_now()
{
    if (!failures_left) return false;
    --failures_left;
    ++injections;
    return true;
}
VKAPI_ATTR VkResult VKAPI_CALL observe_allocate(VkDevice device, const VkMemoryAllocateInfo* info,
                                               const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory)
{
    const auto result = allocate_original(device, info, callbacks, memory);
    if (result == VK_SUCCESS) ++memory_created;
    return result;
}
VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice device, const VkMemoryAllocateInfo* info,
                                       const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory)
{
    if (fail_now()) { *memory = 0; return injected; }
    return observe_allocate(device, info, callbacks, memory);
}
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice device, const VkBufferCreateInfo* info,
                                            const VkAllocationCallbacks* callbacks, VkBuffer* buffer)
{
    const auto result = create_buffer_original(device, info, callbacks, buffer);
    if (result == VK_SUCCESS) ++buffers_created;
    return result;
}
VKAPI_ATTR VkResult VKAPI_CALL fail_create_buffer(VkDevice device, const VkBufferCreateInfo* info,
                                                 const VkAllocationCallbacks* callbacks, VkBuffer* buffer)
{
    if (fail_now()) { *buffer = 0; return injected; }
    return create_buffer(device, info, callbacks, buffer);
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* callbacks)
{
    if (buffer) ++buffers_destroyed;
    destroy_buffer_original(device, buffer, callbacks);
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks)
{
    if (memory) ++memory_destroyed;
    free_memory_original(device, memory, callbacks);
}
VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset,
                                  VkDeviceSize size, VkMemoryMapFlags flags, void** pointer)
{
    if (fail_now()) { *pointer = reinterpret_cast<void*>(1); return injected; }
    return map_original(device, memory, offset, size, flags, pointer);
}
VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice device, const VkCommandPoolCreateInfo* info,
                                          const VkAllocationCallbacks* callbacks, VkCommandPool* pool)
{
    if (fail_now()) { *pool = 0; return injected; }
    return create_pool_original(device, info, callbacks, pool);
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice device, const VkFenceCreateInfo* info,
                                           const VkAllocationCallbacks* callbacks, VkFence* fence)
{
    if (fail_now()) { *fence = 0; return injected; }
    return create_fence_original(device, info, callbacks, fence);
}
VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, uint32_t count, const VkSubmitInfo* info, VkFence fence)
{
    if (fail_now()) return injected;
    return submit_original(queue, count, info, fence);
}
VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice device, uint32_t count, const VkFence* fences,
                                   VkBool32 all, uint64_t timeout)
{
    if (fail_now()) return injected;
    return wait_original(device, count, fences, all, timeout);
}

template<class Function> class Override
{
public:
    Override(Function& slot, Function replacement) : slot_(slot), original_(slot) { slot_ = replacement; }
    ~Override() { slot_ = original_; }
private:
    Function& slot_;
    Function original_;
};

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}
enum class Expected { Oom, Fatal };
template<class Function> void expect(const char* name, VkResult result, Expected expected, Function function)
{
    injected = result;
    failures_left = 1;
    const auto previous = injections;
    bool caught = false;
    try { function(); }
    catch (const std::bad_alloc&) { caught = expected == Expected::Oom; }
    catch (const std::runtime_error&) { caught = expected == Expected::Fatal; }
    failures_left = 0;
    require(caught, "Incorrect or missing Vulkan failure exception");
    require(injections == previous + 1, "Injection did not reach its original Vulkan entry point");
    require(buffers_created == buffers_destroyed, "Failed buffer allocation leaked a Vulkan buffer");
    require(memory_created == memory_destroyed, "Failed buffer allocation leaked Vulkan memory");
    std::cout << name << " passed\n";
}

void run(const ncnn::VulkanDevice* device)
{
    allocate_original = ncnn::vkAllocateMemory;
    create_buffer_original = ncnn::vkCreateBuffer;
    destroy_buffer_original = ncnn::vkDestroyBuffer;
    free_memory_original = ncnn::vkFreeMemory;
    map_original = ncnn::vkMapMemory;
    create_pool_original = ncnn::vkCreateCommandPool;
    create_fence_original = ncnn::vkCreateFence;
    submit_original = ncnn::vkQueueSubmit;
    wait_original = ncnn::vkWaitForFences;
    Override allocation_watch(ncnn::vkAllocateMemory, observe_allocate);
    Override buffer_watch(ncnn::vkCreateBuffer, create_buffer);
    Override destroy_watch(ncnn::vkDestroyBuffer, destroy_buffer);
    Override free_watch(ncnn::vkFreeMemory, free_memory);

    const auto blob = [device] {
        ncnn::VkBlobAllocator allocator(device, 256);
        auto* pointer = allocator.fastMalloc(64);
        if (pointer) allocator.fastFree(pointer);
    };
    const auto weight = [device] {
        ncnn::VkWeightAllocator allocator(device, false, 256);
        auto* pointer = allocator.fastMalloc(64);
        if (pointer) allocator.fastFree(pointer);
    };
    {
        Override failing_allocate(ncnn::vkAllocateMemory, allocate);
        expect("device OOM allocation", VK_ERROR_OUT_OF_DEVICE_MEMORY, Expected::Oom, blob);
        expect("host OOM allocation", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom, weight);
        expect("allocation device loss", VK_ERROR_DEVICE_LOST, Expected::Fatal, blob);
        expect("non-OOM allocation", VK_ERROR_INITIALIZATION_FAILED, Expected::Fatal, blob);
    }
    {
        Override fail_buffer(ncnn::vkCreateBuffer, fail_create_buffer);
        expect("buffer creation OOM", VK_ERROR_OUT_OF_DEVICE_MEMORY, Expected::Oom, weight);
    }
    {
        Override mapping(ncnn::vkMapMemory, map);
        const auto before_free = memory_destroyed;
        const auto staging = [device] {
            ncnn::VkWeightStagingAllocator allocator(device);
            auto* pointer = allocator.fastMalloc(64);
            if (pointer) allocator.fastFree(pointer);
        };
        expect("mapping OOM cleanup", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom, staging);
        require(memory_destroyed == before_free + 1, "Failed mapping did not free its allocation");
    }
    {
        Override pool(ncnn::vkCreateCommandPool, create_pool);
        expect("command constructor OOM", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom,
               [device] { ncnn::VkCompute command(device); });
        expect("command constructor fatal", VK_ERROR_INITIALIZATION_FAILED, Expected::Fatal,
               [device] { ncnn::VkTransfer command(device); });
    }
    {
        Override fence(ncnn::vkCreateFence, create_fence);
        expect("partial command constructor OOM", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom,
               [device] { ncnn::VkTransfer command(device); });
    }
    {
        Override submitting(ncnn::vkQueueSubmit, submit);
        expect("compute submit OOM", VK_ERROR_OUT_OF_DEVICE_MEMORY, Expected::Oom,
               [device] { ncnn::VkCompute command(device); command.submit_and_wait(); });
        expect("transfer submit OOM", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom,
               [device] { ncnn::VkTransfer command(device); command.submit_and_wait(); });
        expect("submit device loss", VK_ERROR_DEVICE_LOST, Expected::Fatal,
               [device] { ncnn::VkCompute command(device); command.submit_and_wait(); });
    }
    {
        Override waiting(ncnn::vkWaitForFences, wait);
        expect("compute wait OOM drains queue", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom,
               [device] { ncnn::VkCompute command(device); command.submit_and_wait(); });
        expect("transfer wait OOM drains queue", VK_ERROR_OUT_OF_HOST_MEMORY, Expected::Oom,
               [device] { ncnn::VkTransfer command(device); command.submit_and_wait(); });
    }
    failures_left = 0;
    std::atomic<bool> ok{true};
    const auto compute = [&] {
        try { for (int i = 0; i < 8; ++i) { ncnn::VkCompute command(device); if (command.submit_and_wait()) ok = false; } }
        catch (...) { ok = false; }
    };
    const auto transfer = [&] {
        try { for (int i = 0; i < 8; ++i) { ncnn::VkTransfer command(device); if (command.submit_and_wait()) ok = false; } }
        catch (...) { ok = false; }
    };
    std::thread first(compute), second(transfer);
    first.join(); second.join();
    require(ok, "Concurrent compute/transfer submission lost a queue after failures");
    std::cout << "concurrent queue reuse passed\n";
}
} // namespace

int main()
{
    if (ncnn::create_gpu_instance() != 0) return 77;
    int result = 0;
    try
    {
        if (ncnn::get_gpu_count() == 0) result = 77;
        else
        {
            const auto* device = ncnn::get_gpu_device();
            if (!device || !device->is_valid()) result = 77;
            else run(device);
        }
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; result = 1; }
    ncnn::destroy_gpu_instance();
    return result;
}

// SPDX-License-Identifier: MIT
#include "vulkan_memory.h"
#include <algorithm>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ernie {
ActivationMemory parse_activation_memory(const std::string& value)
{
    if (value == "auto") return ActivationMemory::Auto;
    if (value == "device") return ActivationMemory::Device;
    if (value == "host") return ActivationMemory::Host;
    throw std::invalid_argument("GPU memory must be auto, device or host");
}

void check_ncnn_memory(int result, const char* action)
{
    if (result == 0) return;
    const std::string message = std::string(action) + " failed (ncnn " + std::to_string(result) + ")";
    if (result == -100) throw GpuAllocationError(message);
    throw std::runtime_error(message);
}

void check_vulkan_memory(int result, const char* action)
{
    if (result == 0) return;
    const std::string message = std::string(action) + " failed (Vulkan " + std::to_string(result) + ")";
    // These numeric values also permit testing the classification in a CPU
    // build. Device loss (-4), invalid usage and other errors are not retried.
    if (result == -1 || result == -2) throw GpuAllocationError(message);
    throw std::runtime_error(message);
}

#if NCNN_VULKAN
namespace {
constexpr auto buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

struct BufferResource
{
    explicit BufferResource(const ncnn::VulkanDevice* value) : device(value) {}
    ~BufferResource()
    {
        if (mapped) ncnn::vkUnmapMemory(device->vkdevice(), memory);
        if (buffer) ncnn::vkDestroyBuffer(device->vkdevice(), buffer, nullptr);
        if (memory) ncnn::vkFreeMemory(device->vkdevice(), memory, nullptr);
    }
    const ncnn::VulkanDevice* device;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    void disown() { buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; mapped = nullptr; }
};

void create_storage_buffer(BufferResource& out, size_t size)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = buffer_usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const auto result = ncnn::vkCreateBuffer(out.device->vkdevice(), &info, nullptr, &out.buffer);
    if (result != VK_SUCCESS) out.buffer = VK_NULL_HANDLE;
    check_vulkan_memory(result, "Create activation buffer");
}

// Select the same type as VkBlobAllocator using a real storage buffer's
// memoryTypeBits. This needs no device allocation when VRAM is already tight.
std::uint32_t compute_memory_type(const ncnn::VulkanDevice* device)
{
    BufferResource probe(device);
    create_storage_buffer(probe, 256);
    VkMemoryRequirements requirements{};
    ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), probe.buffer, &requirements);
    const auto& properties = device->info.physicalDeviceMemoryProperties();
    auto selected = device->find_memory_index(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        device->info.type() == 1 ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 0,
        device->info.type() == 1 ? 0 : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (device->info.type() == 1)
    {
        const auto local = device->find_memory_index(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (selected < properties.memoryTypeCount && local < properties.memoryTypeCount)
        {
            const auto selected_heap = properties.memoryTypes[selected].heapIndex;
            const auto local_heap = properties.memoryTypes[local].heapIndex;
            if (local_heap < selected_heap && properties.memoryHeaps[local_heap].size >
                    properties.memoryHeaps[selected_heap].size)
                selected = local;
        }
    }
    if (selected >= properties.memoryTypeCount)
        throw std::runtime_error("No compatible Vulkan compute memory type");
    return selected;
}

size_t aligned_buffer_size(const ncnn::VulkanDevice* device, size_t size)
{
    size_t alignment = std::max<size_t>(16, device->info.buffer_offset_alignment());
    if (device->info.type() == 1)
        alignment = std::max({alignment, device->info.memory_map_alignment(), device->info.non_coherent_atom_size()});
    if (device->info.support_VK_KHR_robustness2() || device->info.support_VK_EXT_robustness2())
        alignment = std::max<size_t>(alignment,
            device->info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment);
    if (!size || size > std::numeric_limits<size_t>::max() - (alignment - 1))
        throw std::invalid_argument("Invalid activation buffer size");
    return (size + alignment - 1) / alignment * alignment;
}
} // namespace

WeightPlacement::BudgetReader compute_memory_budget_reader(const ncnn::VulkanDevice* device)
{
    if (!device) throw std::invalid_argument("Compute memory query requires a Vulkan device");
    return vulkan_memory_budget_reader(device, compute_memory_type(device));
}

struct AdaptiveVkAllocator::Impl
{
    struct HostBlock
    {
        std::uint64_t bytes;
        bool retired = false;
    };
    const ncnn::VulkanDevice* device;
    ActivationMemory mode;
    std::uint64_t gpu_reserve, host_limit, ram_reserve;
    HostAvailableReader host_reader;
    BudgetReader budget_reader;
    // No implicit 16 MiB growth: before a new block the budget test must be
    // about the requested allocation. Released device regions still reuse.
    ncnn::VkBlobAllocator device_pool;
    VulkanMemoryStats counters;
    std::unordered_map<ncnn::VkBufferMemory*, HostBlock> host_blocks;
    std::unordered_set<ncnn::VkBufferMemory*> device_buffers;
    std::unordered_set<ncnn::VkImageMemory*> device_images;

    Impl(const ncnn::VulkanDevice* value, ActivationMemory memory,
        std::uint64_t gpu, std::uint64_t host, std::uint64_t ram,
        HostAvailableReader reader, BudgetReader budget)
        : device(value), mode(memory), gpu_reserve(gpu), host_limit(host), ram_reserve(ram),
          host_reader(std::move(reader)), budget_reader(std::move(budget)), device_pool(value, 0)
    {
        if (mode != ActivationMemory::Host)
        {
            device_pool.buffer_memory_type_index = compute_memory_type(device);
            device_pool.mappable = device->is_mappable(device_pool.buffer_memory_type_index);
            device_pool.coherent = device->is_coherent(device_pool.buffer_memory_type_index);
            if (!budget_reader)
                budget_reader = vulkan_memory_budget_reader(device, device_pool.buffer_memory_type_index);
        }
    }

    bool device_fits(std::uint64_t size) const
    {
        if (mode != ActivationMemory::Auto || !budget_reader) return true;
        const auto budget = budget_reader();
        if (!budget) return true; // Missing GPU telemetry still permits a real allocation attempt.
        const auto available = budget->usage_bytes >= budget->budget_bytes ? 0 :
            budget->budget_bytes - budget->usage_bytes;
        return available >= gpu_reserve && size <= available - gpu_reserve;
    }

    void require_host_budget(std::uint64_t size) const
    {
        if (counters.host_live_bytes > host_limit || size > host_limit - counters.host_live_bytes)
            throw GpuAllocationError("Activation RAM budget exceeded");
        const auto available = host_reader ? host_reader() : std::nullopt;
        if (!available)
            throw GpuAllocationError("Activation RAM headroom is unavailable");
        if (*available < ram_reserve || size > *available - ram_reserve)
            throw GpuAllocationError("Activation RAM reserve would be exceeded");
    }

    ncnn::VkBufferMemory* allocate_host(size_t size)
    {
        require_host_budget(size);
        BufferResource resource(device);
        create_storage_buffer(resource, size);
        VkMemoryRequirements requirements{};
        ncnn::vkGetBufferMemoryRequirements(device->vkdevice(), resource.buffer, &requirements);
        // Driver alignment/padding is part of both limits, not just the tensor.
        require_host_budget(requirements.size);
        const auto type = device->find_memory_index(requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const auto& properties = device->info.physicalDeviceMemoryProperties();
        if (type >= properties.memoryTypeCount)
            throw GpuAllocationError("No coherent host-visible Vulkan activation memory type");

        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        const auto allocation_result = ncnn::vkAllocateMemory(device->vkdevice(), &allocation, nullptr, &resource.memory);
        if (allocation_result != VK_SUCCESS) resource.memory = VK_NULL_HANDLE;
        check_vulkan_memory(allocation_result, "Allocate activation RAM");
        check_vulkan_memory(ncnn::vkBindBufferMemory(device->vkdevice(), resource.buffer, resource.memory, 0),
                            "Bind activation RAM");
        const auto map_result = ncnn::vkMapMemory(device->vkdevice(), resource.memory, 0, requirements.size, 0,
                                                &resource.mapped);
        if (map_result != VK_SUCCESS) resource.mapped = nullptr;
        check_vulkan_memory(map_result, "Map activation RAM");
        auto ptr = std::make_unique<ncnn::VkBufferMemory>();
        ptr->buffer = resource.buffer;
        ptr->memory = resource.memory;
        ptr->mapped_ptr = resource.mapped;
        ptr->offset = 0;
        ptr->capacity = size;
        ptr->memory_type_index = type;
        ptr->access_flags = 0;
        ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        ptr->refcount = 0; // VkMat initializes its own reference on adoption.
        host_blocks.emplace(ptr.get(), HostBlock{requirements.size, false});
        resource.disown();
        counters.host_live_bytes += requirements.size;
        counters.host_peak_bytes = std::max(counters.host_peak_bytes, counters.host_live_bytes);
        ++counters.host_allocations;
        if (properties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            ++counters.host_device_local_allocations;
        else
            ++counters.host_non_device_local_allocations;
        return ptr.release();
    }

    void destroy_host(ncnn::VkBufferMemory* ptr, std::uint64_t bytes)
    {
        ncnn::vkUnmapMemory(device->vkdevice(), ptr->memory);
        ncnn::vkDestroyBuffer(device->vkdevice(), ptr->buffer, nullptr);
        ncnn::vkFreeMemory(device->vkdevice(), ptr->memory, nullptr);
        counters.host_live_bytes -= bytes;
        delete ptr;
    }
};

AdaptiveVkAllocator::AdaptiveVkAllocator(const ncnn::VulkanDevice* device, ActivationMemory mode,
    std::uint64_t gpu_reserve_bytes, std::uint64_t host_limit_bytes, std::uint64_t ram_reserve_bytes,
    HostAvailableReader reader, BudgetReader budget_reader)
    : ncnn::VkAllocator(device)
{
    if (!device) throw std::invalid_argument("Activation allocator requires a Vulkan device");
    if (mode != ActivationMemory::Auto && mode != ActivationMemory::Device && mode != ActivationMemory::Host)
        throw std::invalid_argument("Invalid activation memory mode");
    impl_ = std::make_unique<Impl>(device, mode, gpu_reserve_bytes, host_limit_bytes, ram_reserve_bytes,
                                 std::move(reader), std::move(budget_reader));
    buffer_memory_type_index = impl_->device_pool.buffer_memory_type_index;
    // ncnn has one mappable flag per allocator, but our buffers may mix types.
    // Keep uploads/downloads on its staging path; the per-buffer overrides
    // below still implement correct flush/invalidate for coherent host memory.
    mappable = false;
    coherent = false;
}

AdaptiveVkAllocator::~AdaptiveVkAllocator()
{
    // Same lifetime requirement as ncnn's pools: callers synchronize commands
    // and destroy all tensors before the allocator. Never wait on every free.
    for (const auto& block : impl_->host_blocks)
        impl_->destroy_host(block.first, block.second.bytes);
}

VulkanMemoryStats AdaptiveVkAllocator::stats() const { return impl_->counters; }

ncnn::VkBufferMemory* AdaptiveVkAllocator::fastMalloc(size_t size)
{
    const auto aligned_size = aligned_buffer_size(vkdev, size);
    bool host = impl_->mode == ActivationMemory::Host;
    if (!host && impl_->device_fits(aligned_size))
    {
        ncnn::VkBufferMemory* ptr = nullptr;
        try { ptr = impl_->device_pool.fastMalloc(aligned_size); }
        catch (const std::bad_alloc&) { /* Authenticated ncnn Vulkan OOM propagation. */ }
        catch (const GpuAllocationError&) { /* Same typed contract for caller extensions. */ }
        if (ptr)
        {
            try { impl_->device_buffers.insert(ptr); }
            catch (...) { impl_->device_pool.fastFree(ptr); throw; }
            ++impl_->counters.device_allocations;
            buffer_memory_type_index = ptr->memory_type_index;
            return ptr;
        }
        // Patched ncnn preserves Vulkan OOM as bad_alloc. A nullptr retains its
        // existing -100 allocation convention. Other errors, including device
        // loss, propagate out of the narrow allocation catch above.
        ++impl_->counters.allocation_failures;
        if (impl_->mode == ActivationMemory::Device)
            throw GpuAllocationError("Vulkan device activation allocation failed");
        host = true;
    }
    else if (!host)
        host = true;
    if (impl_->mode == ActivationMemory::Auto) ++impl_->counters.fallbacks;
    try { return impl_->allocate_host(aligned_size); }
    catch (const GpuAllocationError&) { ++impl_->counters.allocation_failures; throw; }
}

void AdaptiveVkAllocator::fastFree(ncnn::VkBufferMemory* ptr)
{
    if (!ptr) return;
    const auto host = impl_->host_blocks.find(ptr);
    if (host != impl_->host_blocks.end())
    {
        host->second.retired = true;
        return;
    }
    if (impl_->device_buffers.erase(ptr)) impl_->device_pool.fastFree(ptr);
}

int AdaptiveVkAllocator::flush(ncnn::VkBufferMemory* ptr)
{
    return impl_->host_blocks.count(ptr) ? 0 : impl_->device_pool.flush(ptr);
}
int AdaptiveVkAllocator::invalidate(ncnn::VkBufferMemory* ptr)
{
    return impl_->host_blocks.count(ptr) ? 0 : impl_->device_pool.invalidate(ptr);
}

ncnn::VkImageMemory* AdaptiveVkAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
{
    if (w <= 0 || h <= 0 || c <= 0 || (elempack != 1 && elempack != 4 && elempack != 8) ||
            !elemsize || elemsize % elempack ||
            (elemsize / elempack != 1 && elemsize / elempack != 2 && elemsize / elempack != 4))
        throw std::invalid_argument("Invalid Vulkan image allocation shape");
    const auto max_dimension = vkdev->info.max_image_dimension_3d();
    if (static_cast<std::uint64_t>(w) * (elempack == 8 ? 2u : 1u) > max_dimension ||
        static_cast<std::uint64_t>(h) > max_dimension || static_cast<std::uint64_t>(c) > max_dimension)
        throw std::invalid_argument("Vulkan image dimensions exceed device limits");
    auto* ptr = impl_->device_pool.fastMalloc(w, h, c, elemsize, elempack);
    if (!ptr)
    {
        ++impl_->counters.allocation_failures;
        throw GpuAllocationError("Vulkan image allocation failed; image storage cannot spill to RAM");
    }
    try { impl_->device_images.insert(ptr); }
    catch (...) { impl_->device_pool.fastFree(ptr); throw; }
    image_memory_type_index = ptr->memory_type_index;
    ++impl_->counters.device_allocations;
    return ptr;
}

void AdaptiveVkAllocator::fastFree(ncnn::VkImageMemory* ptr)
{
    if (ptr && impl_->device_images.erase(ptr)) impl_->device_pool.fastFree(ptr);
}

void AdaptiveVkAllocator::reclaim_completed()
{
    for (auto block = impl_->host_blocks.begin(); block != impl_->host_blocks.end();)
    {
        if (!block->second.retired) { ++block; continue; }
        impl_->destroy_host(block->first, block->second.bytes);
        block = impl_->host_blocks.erase(block);
    }
}

void AdaptiveVkAllocator::clear()
{
    reclaim_completed();
    if (impl_->device_buffers.empty() && impl_->device_images.empty()) impl_->device_pool.clear();
}

void reclaim_completed(ncnn::VkAllocator* allocator)
{
    if (auto* adaptive = dynamic_cast<AdaptiveVkAllocator*>(allocator)) adaptive->reclaim_completed();
}
#endif
} // namespace ernie

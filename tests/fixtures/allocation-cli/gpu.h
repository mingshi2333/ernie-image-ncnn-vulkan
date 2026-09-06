// Test-only non-creating Vulkan instance query. No GPU API is linked.
#pragma once
#include <cstdint>
#define VK_NULL_HANDLE 0
namespace ncnn { std::uint64_t get_gpu_instance(); }

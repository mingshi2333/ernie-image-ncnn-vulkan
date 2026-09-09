// SPDX-License-Identifier: MIT
#pragma once
#include "host_memory.h"
#include <cstdint>
#include <functional>

namespace ernie {
// One generation attempt. Background preparation never mutates this object.
struct MemoryExecution
{
    std::uint64_t prefetch_bytes = 0;
    std::uint64_t ram_reserve_bytes = 3ull * 1024 * 1024 * 1024;
    int query_rows = 128;
    std::function<void(int step)> before_step;
    HostAvailableReader available = host_memory_available_reader();
};
} // namespace ernie

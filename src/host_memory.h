// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>

namespace ernie {
using HostAvailableReader = std::function<std::optional<std::uint64_t>()>;

// Linux input paths are explicit for deterministic reader tests. Production
// calls the zero-argument function; these paths are not runtime user options.
struct HostMemoryFiles
{
    std::filesystem::path meminfo = "/proc/meminfo";
    std::filesystem::path membership = "/proc/self/cgroup";
    std::filesystem::path cgroup_root = "/sys/fs/cgroup";
};
HostAvailableReader linux_host_memory_available_reader(HostMemoryFiles);

// Estimated allocatable headroom. Linux uses MemAvailable and every finite
// cgroup-v2 ancestor. Credit at most half the clean active + inactive file pages;
// they are not equivalent to anonymous or pinned weights. No swap is credited.
// Missing optional file statistics give no reclaim credit. Windows bounds
// available physical RAM by remaining process commit and virtual address space.
// Windows Job membership and Wine leave the optional cache disabled until their
// full enclosing limits can be established. Streaming generation is unaffected.
// Required query failures and unsupported platforms disable cache admission.
HostAvailableReader host_memory_available_reader();
} // namespace ernie

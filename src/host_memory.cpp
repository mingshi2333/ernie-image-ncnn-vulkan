// SPDX-License-Identifier: MIT
#include "host_memory.h"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ernie {
namespace {
using Bytes = std::uint64_t;
std::optional<Bytes> unsigned_value(const std::string& text)
{
    if (text.empty()) return std::nullopt;
    Bytes value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}
std::optional<std::string> single_value(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::string value, extra;
    if (!(file >> value) || file >> extra) return std::nullopt;
    return value;
}
Bytes subtract(Bytes value, Bytes deduction) { return value > deduction ? value - deduction : 0; }

std::optional<Bytes> physical_available(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::string line;
    std::optional<Bytes> result;
    while (std::getline(file, line))
    {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream row(line.substr(13));
        std::string number, unit, extra;
        if (result || !(row >> number >> unit) || unit != "kB" || row >> extra) return std::nullopt;
        const auto kib = unsigned_value(number);
        if (!kib || *kib > UINT64_MAX / 1024) return std::nullopt;
        result = *kib * 1024;
    }
    return result;
}

Bytes inactive_file_credit(const std::filesystem::path& path, Bytes usage)
{
    // Linux memory.stat documents list-based inactive_file separately from
    // type-based file/shmem/dirty/writeback counters. Bound by both the list
    // and non-shmem file total, then exclude dirty/writeback pages. Anonymous
    // and unevictable pages live on other lists; subtracting their totals again
    // would wrongly charge pinned weights against unrelated clean file pages.
    std::map<std::string, std::optional<Bytes>> fields{{"file", {}}, {"inactive_file", {}},
        {"file_dirty", {}}, {"file_writeback", {}}, {"shmem", {}}};
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream row(line);
        std::string key, number, extra;
        if (!(row >> key)) continue;
        auto field = fields.find(key);
        if (field == fields.end()) continue;
        if (field->second || !(row >> number) || row >> extra) return 0;
        field->second = unsigned_value(number);
        if (!field->second) return 0;
    }
    for (const auto& field : fields) if (!field.second) return 0;
    Bytes credit = std::min({usage, subtract(*fields["file"], *fields["shmem"]), *fields["inactive_file"]});
    for (const auto* key : {"file_dirty", "file_writeback"})
        credit = subtract(credit, *fields[key]);
    return credit;
}

std::optional<std::filesystem::path> membership_path(const HostMemoryFiles& files)
{
    std::ifstream membership(files.membership);
    std::string line;
    std::optional<std::filesystem::path> path;
    while (std::getline(membership, line))
    {
        if (line.rfind("0::/", 0) != 0) continue;
        if (path) return std::nullopt;
        const auto relative = std::filesystem::path(line.substr(3)).relative_path();
        for (const auto& part : relative)
            if (part == ".." || part == ".") return std::nullopt;
        path = files.cgroup_root / relative;
    }
    return path;
}
} // namespace

HostAvailableReader linux_host_memory_available_reader(HostMemoryFiles files)
{
    return [files = std::move(files)]() -> std::optional<Bytes> {
        auto available = physical_available(files.meminfo);
        auto member = membership_path(files); // Recheck if the process migrated.
        if (!available || !member) return std::nullopt;
        for (auto path = *member;; path = path.parent_path())
        {
            std::error_code error;
            if (!std::filesystem::is_directory(path, error) || error) return std::nullopt;
            const bool has_limit = std::filesystem::exists(path / "memory.max", error);
            if (error) return std::nullopt;
            if (has_limit)
            {
                const auto limit = single_value(path / "memory.max");
                if (!limit) return std::nullopt;
                if (*limit != "max")
                {
                    const auto maximum = unsigned_value(*limit);
                    const auto before_text = single_value(path / "memory.current");
                    if (!maximum || !before_text) return std::nullopt;
                    const auto before = unsigned_value(*before_text);
                    if (!before) return std::nullopt;
                    const auto credit = inactive_file_credit(path / "memory.stat", *before);
                    const auto after_text = single_value(path / "memory.current");
                    if (!after_text) return std::nullopt;
                    const auto after = unsigned_value(*after_text);
                    if (!after) return std::nullopt;
                    // Counters are not an atomic snapshot. Use the larger
                    // usage observed around the stat read and saturating math.
                    const auto working = subtract(std::max(*before, *after), credit);
                    *available = std::min(*available, subtract(*maximum, working));
                }
            }
            else
            {
                // A root or disabled controller may have neither interface;
                // a partial/disappearing controller must not hide its limit.
                const bool has_usage = std::filesystem::exists(path / "memory.current", error);
                if (error || has_usage) return std::nullopt;
            }
            if (path == files.cgroup_root) return available;
            if (path == path.parent_path()) return std::nullopt;
        }
    };
}

HostAvailableReader host_memory_available_reader()
{
#if defined(__linux__)
    return linux_host_memory_available_reader({});
#else
    return []() -> std::optional<Bytes> { return std::nullopt; };
#endif
}
} // namespace ernie

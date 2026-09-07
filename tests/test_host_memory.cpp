// SPDX-License-Identifier: MIT
#include "host_memory.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace fs = std::filesystem;
using Bytes = std::uint64_t;
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
void write(const fs::path& path, const std::string& text)
{
    std::ofstream file(path);
    file << text;
    require(bool(file), "Cannot write host-memory fixture");
}
std::string stats(Bytes file = 800, Bytes inactive = 630, Bytes dirty = 20,
                  Bytes writeback = 10, Bytes shmem = 40, Bytes unevictable = 30,
                  Bytes active = 0)
{
    return "file " + std::to_string(file) + "\ninactive_file " + std::to_string(inactive) +
        "\nfile_dirty " + std::to_string(dirty) + "\nfile_writeback " + std::to_string(writeback) +
        "\nshmem " + std::to_string(shmem) + "\nunevictable " + std::to_string(unevictable) +
        "\nactive_file " + std::to_string(active) +
        "\nslab_reclaimable 9999\nanon 9999\nswapcached 9999\nfuture_counter 123\n";
}
struct Fixture
{
    fs::path root = fs::temp_directory_path() /
        ("ernie-host-memory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ernie::HostMemoryFiles files{root / "meminfo", root / "membership", root / "cgroup"};
    fs::path parent = files.cgroup_root / "parent", leaf = parent / "leaf";
    Fixture()
    {
        require(fs::create_directories(leaf), "Cannot create fixture directory");
        write(files.meminfo, "MemAvailable: 1000 kB\nSwapFree: 999999 kB\n");
        write(files.membership, "0::/parent/leaf\n");
        write(leaf / "memory.max", "1000\n");
        write(leaf / "memory.current", "900\n");
        write(leaf / "memory.stat", stats());
    }
    ~Fixture() { std::error_code error; fs::remove_all(root, error); }
};
void check(const ernie::HostAvailableReader& reader, std::optional<Bytes> expected, const char* why)
{
    const auto actual = reader();
    if (actual != expected)
        throw std::runtime_error(std::string(why) + ": got " +
            (actual ? std::to_string(*actual) : "unavailable"));
}
} // namespace

int main()
{
    try
    {
        Fixture f;
        auto reader = ernie::linux_host_memory_available_reader(f.files);
        // Retain half of 600 clean file bytes: 100 hard headroom + 300 credit.
        // Moving the same pages between LRUs must not change admission.
        check(reader, 400, "Clean file-cache headroom was lost");
        for (const Bytes active : {Bytes(0), Bytes(315), Bytes(630)})
        {
            write(f.leaf / "memory.stat", stats(800, 630 - active, 20, 10, 40, 30, active));
            check(reader, 400, "File LRU movement changed available headroom");
        }
        write(f.leaf / "memory.stat", stats(800, 630, 20, 10, 40, UINT64_MAX));
        check(reader, 400, "Unrelated pinned pages were deducted from the file list again");
        write(f.leaf / "memory.stat", stats(800, 700, 0, 0, 750, 0));
        check(reader, 125, "Shared memory was credited as clean file pages");
        write(f.leaf / "memory.stat", stats(900, 150, 0, 0, 0, 0));
        check(reader, 175, "Credit exceeded half the file LRU lists");
        write(f.leaf / "memory.stat", stats(100, 700, 0, 0, 0, 0));
        check(reader, 150, "Credit exceeded half the file bytes");
        write(f.leaf / "memory.stat", stats(800, 0, 0, 0, 0, 0, 0));
        check(reader, 100, "File bytes outside reclaim lists were credited");
        write(f.leaf / "memory.stat", stats(101, 50, 0, 0, 0, 0, 51));
        check(reader, 150, "Odd file totals did not retain at least half");
        write(f.leaf / "memory.stat", stats(800, UINT64_MAX, 0, 0, 0, 0, UINT64_MAX));
        check(reader, 500, "File list sum overflowed");
        write(f.leaf / "memory.stat", stats(800, 0, 799, 1, 0, 0, 800));
        check(reader, 100, "Dirty active-file pages were credited");
        write(f.leaf / "memory.stat", stats(800, 700, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX));
        check(reader, 100, "Excluded categories underflowed or became reclaim credit");
        write(f.leaf / "memory.stat", stats(0, 0, 0, 0, 0, 0));
        write(f.leaf / "memory.current", "970\n");
        check(reader, 30, "Anonymous pressure was hidden");
        write(f.leaf / "memory.current", "1001\n");
        check(reader, 0, "Over-limit usage underflowed");
        write(f.leaf / "memory.current", "900\n");
        write(f.leaf / "memory.stat", stats());
        write(f.parent / "memory.max", "600\n");
        write(f.parent / "memory.current", "500\n");
        write(f.parent / "memory.stat", stats(0, 0, 0, 0, 0, 0));
        check(reader, 100, "An ancestor's limit was bypassed");
        write(f.parent / "memory.max", "max\n");
        check(reader, 400, "An updated parent limit was not read");
        write(f.files.meminfo, "MemAvailable: 0 kB\n");
        check(reader, 0, "Host physical availability was bypassed");
        write(f.files.meminfo, "MemAvailable: 1000 kB\n");
        fs::remove(f.leaf / "memory.stat");
        check(reader, 100, "Missing optional statistics invented credit");
        for (const auto& invalid : {stats() + "file 999\n", stats() + "file 0\n",
                                   stats() + "active_file 0\n",
                                   std::string("file 800\ninactive_file 630\nfile_dirty 20\nfile_writeback 10\nshmem 40\n"),
                                   std::string("inactive_file 800\n"), std::string("file -1\n")})
        {
            write(f.leaf / "memory.stat", invalid);
            check(reader, 100, "Corrupt/incomplete statistics invented credit");
        }
        write(f.leaf / "memory.stat", stats());
        for (const auto& invalid : {"-1", "123x", "18446744073709551616", "1000 2000", ""})
        {
            write(f.leaf / "memory.max", invalid);
            check(reader, std::nullopt, "Malformed limit accepted");
            write(f.leaf / "memory.max", "1000");
            write(f.leaf / "memory.current", invalid);
            check(reader, std::nullopt, "Malformed current usage accepted");
            write(f.leaf / "memory.current", "900");
        }
        write(f.leaf / "memory.max", "18446744073709551615\n");
        write(f.leaf / "memory.current", "18446744073709551610\n");
        write(f.leaf / "memory.stat", stats(0, 0, 0, 0, 0, 0));
        check(reader, 5, "Large counters overflowed");
        write(f.leaf / "memory.stat", stats(UINT64_MAX, UINT64_MAX, 0, 0, 0, 0));
        check(reader, 1024000, "Credit arithmetic overflowed or exceeded host availability");
        write(f.leaf / "memory.max", "max\n");
        check(reader, 1024000, "Unlimited cgroups changed host availability");
        for (const auto& invalid : {"", "MemAvailable: -1 kB\n", "MemAvailable: 1 MB\n",
            "MemAvailable: 18014398509481984 kB\n", "MemAvailable: 0 kB\nMemAvailable: 1 kB\n"})
        {
            write(f.files.meminfo, invalid);
            check(reader, std::nullopt, "Malformed host availability accepted");
        }
        write(f.files.meminfo, "MemAvailable: 1000 kB\n");
        for (const auto& invalid : {"1:memory:/parent/leaf\n", "0::/../outside\n",
                                   "0::/parent/leaf\n0::/parent\n", "0::/missing\n"})
        {
            write(f.files.membership, invalid);
            check(reader, std::nullopt, "Invalid membership hid a limit");
        }
        write(f.files.membership, "0::/parent\n");
        write(f.parent / "memory.max", "600\n");
        check(reader, 100, "Process migration reused a stale cgroup path");
        fs::remove(f.parent / "memory.max");
        check(reader, std::nullopt, "A partial controller hid its limit");
        fs::remove(f.parent / "memory.current");
        check(reader, 1024000, "A disabled controller was treated as a finite limit");
        write(f.files.membership, "0::/\n");
        check(reader, 1024000, "Root membership did not terminate correctly");
#if defined(__linux__)
        const auto live = ernie::host_memory_available_reader()();
        require(live && *live, "Live Linux memory query unavailable");
        std::cout << "Live Linux available estimate=" << *live << " bytes\n";
#endif
        std::cout << "File LRU movement, bounded credit, real pressure, ancestors, changing inputs, corruption, "
                     "missing counters and overflow contracts pass\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

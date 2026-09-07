// Read-only CPU diagnostic: compare ncnn's batched and vector InnerProduct paths.
#include <net.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

int main(int argc, char** argv)
{
    try
    {
        if (argc != 8) throw std::runtime_error("Usage: runner PARAM BIN INPUT OUTPUT ROWS K N");
        const int rows = std::stoi(argv[5]), k = std::stoi(argv[6]), n = std::stoi(argv[7]);
        if (rows < 1 || rows > 4096 || k < 1 || k > 16384 || n < 1 || n > 16384)
            throw std::runtime_error("Invalid bounded dimensions");
        if (std::filesystem::exists(argv[4])) throw std::runtime_error("Output already exists");
        if (std::filesystem::file_size(argv[3]) != size_t(rows) * k * 4)
            throw std::runtime_error("Input byte count mismatch");
        ncnn::Net net;
        net.opt.num_threads = 2;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
        const auto start = std::chrono::steady_clock::now();
        if (net.load_param(argv[1]) || net.load_model(argv[2])) throw std::runtime_error("Model load failed");
        const auto loaded = std::chrono::steady_clock::now();
        std::ifstream input(argv[3], std::ios::binary);
        std::ofstream output(argv[4], std::ios::binary);
        if (!input || !output) throw std::runtime_error("Cannot open data file");
        // A one-dimensional Mat deliberately dispatches the upstream vector kernel.
        ncnn::Mat row(k);
        for (int r = 0; r < rows; ++r)
        {
            input.read(static_cast<char*>(row.data), size_t(k) * 4);
            if (!input) throw std::runtime_error("Input read failed");
            for (int i = 0; i < k; ++i) if (!std::isfinite(row[i])) throw std::runtime_error("Nonfinite input");
            auto ex = net.create_extractor();
            if (ex.input("in0", row)) throw std::runtime_error("Input failed");
            ncnn::Mat result;
            if (ex.extract("out0", result)) throw std::runtime_error("Extraction failed");
            ncnn::Mat plain;
            ncnn::convert_packing(result, plain, 1, net.opt);
            if (plain.dims != 1 || plain.w != n || plain.elemsize != 4u)
                throw std::runtime_error("Unexpected vector output layout");
            for (int i = 0; i < n; ++i) if (!std::isfinite(plain[i])) throw std::runtime_error("Nonfinite output");
            output.write(static_cast<const char*>(plain.data), size_t(n) * 4);
        }
        output.close();
        if (!output) throw std::runtime_error("Output close failed");
        const auto finished = std::chrono::steady_clock::now();
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS usage{};
        if (!GetProcessMemoryInfo(GetCurrentProcess(), &usage, sizeof(usage)))
            throw std::runtime_error("Peak working-set query failed");
        const auto peak_kib = usage.PeakWorkingSetSize / 1024;
        const char* peak_name = "peak_working_set_kib";
#else
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("RSS query failed");
        const auto peak_kib = usage.ru_maxrss;
        const char* peak_name = "peak_rss_kib";
#endif
        std::cout << "{\"rows\":" << rows << ",\"threads\":2,\"load_seconds\":"
                  << std::chrono::duration<double>(loaded - start).count()
                  << ",\"compute_and_io_seconds\":" << std::chrono::duration<double>(finished - loaded).count()
                  << ",\"" << peak_name << "\":" << peak_kib << "}\n";
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

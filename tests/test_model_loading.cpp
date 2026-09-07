// SPDX-License-Identifier: MIT
#include "model_loading.h"
#include "net.h"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
struct Temporary
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ernie-model-loading-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() { require(std::filesystem::create_directory(path), "Cannot create fixture directory"); }
    ~Temporary() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
bool file_mapped(const std::filesystem::path& path)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
        if (line.find(path.string()) != std::string::npos) return true;
    return false;
}
void exercise(const std::filesystem::path& weights, const std::string& mode, bool build_default)
{
    auto owner = std::make_unique<ncnn::Net>();
    auto& net = *owner;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    net.opt.use_mapped_model_loading = ernie::request_mapped_model_loading(mode, build_default);
    require(net.load_param_mem("7767517\n2 2\nInput input 0 1 in0\n"
        "Gemm projection 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=1 8=2 9=2 10=-1\n") == 0,
        "Cannot load GEMM graph");
    require(net.load_model(weights.string().c_str()) == 0, "Cannot load fixture weights");
#if defined(__linux__)
    require(file_mapped(weights) == net.opt.use_mapped_model_loading,
            "Requested loader did not select the expected mapping path");
#endif
    for (int pass = 1; pass <= 3; ++pass)
    {
        // Extractors end each round while the owning Net keeps its mapping.
        auto ex = net.create_extractor();
        ncnn::Mat input(2, 1), output;
        input[0] = 1.25f * pass; input[1] = -2.f * pass;
        require(ex.input("in0", input) == 0 && ex.extract("out0", output) == 0,
                "CPU GEMM extraction failed");
        require(output.w == 2 && output[0] == 2.5f * pass && output[1] == -6.f * pass,
                "Mapped/stdio weights changed inference or lost their lifetime");
    }
    net.clear();
    // The pinned ncnn owns MappedFile in NetPrivate, so its lifetime ends at
    // Net destruction, not at clear(). WeightSession evicts by destroying Net.
    owner.reset();
#if defined(__linux__)
    require(!file_mapped(weights), "Destroyed Net retained its model mapping");
#endif
}
} // namespace
int main()
{
    try
    {
        Temporary temp;
        for (bool bf16 : {false, true})
        {
            const auto path = temp.path / (bf16 ? "bf16.bin" : "fp32.bin");
            {
                std::ofstream file(path, std::ios::binary);
                const std::uint32_t tag = bf16 ? 0x01348b83 : 0;
                const float fp32[] = {2.f, 0.f, 0.f, 3.f};
                const std::uint16_t packed[] = {0x4000, 0, 0, 0x4040};
                file.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
                if (bf16) file.write(reinterpret_cast<const char*>(packed), sizeof(packed));
                else file.write(reinterpret_cast<const char*>(fp32), sizeof(fp32));
                require(bool(file), "Cannot write fixture weights");
            }
            for (bool build_default : {false, true})
                for (const std::string mode : {"default", "stdio", "mapped"})
                    exercise(path, mode, build_default);
        }
        for (const std::string mode : {"stdio", "mapped"})
        {
            ncnn::Net missing;
            missing.opt.use_vulkan_compute = false;
            missing.opt.use_mapped_model_loading = ernie::request_mapped_model_loading(mode, false);
            require(missing.load_model((temp.path / "absent.bin").string().c_str()) != 0,
                    "Missing model was accepted");
        }
        std::cout << "CPU FP32/BF16 files: both loading modes, build defaults, repeated extraction, "
                     "mapping ownership and missing-file rejection pass\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

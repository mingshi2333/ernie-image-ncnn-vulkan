// SPDX-License-Identifier: MIT
#include "component_files.h"
#include "net.h"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F action)
{
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "Invalid component was accepted");
}
struct Temporary
{
    fs::path path = fs::temp_directory_path() /
        fs::u8path(u8"ernie-component-\u6a21\u578b \U0001f5bc-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() { require(fs::create_directory(path), "Cannot create isolated test directory"); }
    ~Temporary() { std::error_code ignored; fs::remove_all(path, ignored); }
};
ncnn::Mat infer(ncnn::Net &net)
{
    ncnn::Mat input(2); input[0] = 1.25f; input[1] = -2.f;
    auto ex = net.create_extractor();
    ncnn::Mat result;
    require(ex.input("in0", input) == 0 && ex.extract("out0", result) == 0, "Tiny model failed");
    return result;
}
}
int main()
{
    try
    {
        Temporary dir;
        const std::string graph = "7767517\n2 2\nInput input 0 1 in0\n"
                                  "InnerProduct projection 1 1 in0 out0 0=2 1=1 2=4\n";
        const auto param = dir.path / "head.ncnn.param", weights = dir.path / "head.ncnn.bin";
        { std::ofstream file(param, std::ios::binary); file << graph; }
        {
            std::ofstream file(weights, std::ios::binary);
            const uint32_t fp32_tag = 0;
            const float data[] = {2.f, 1.f, 3.f, -1.f, .5f, -.25f};
            file.write(reinterpret_cast<const char *>(&fp32_tag), sizeof(fp32_tag));
            file.write(reinterpret_cast<const char *>(data), sizeof(data));
        }
        ncnn::Net legacy, resolved;
        for (auto *net : {&legacy, &resolved})
        {
            net->opt.use_vulkan_compute = false;
            net->opt.use_packing_layout = false;
            net->opt.num_threads = 1;
        }
        require(legacy.load_param(param.c_str()) == 0 &&
                legacy.load_model(weights.c_str()) == 0, "Legacy model load failed");
        auto files = ernie::component_files(dir.path, "head");
        require(files.param_text == graph && files.weight_path == weights.u8string(), "Descriptor differs");
        // The descriptor owns graph text; no graph pathname is needed at load time.
        fs::remove(param);
        ernie::load_component(resolved, files);
        files = {};
        const auto old = infer(legacy), actual = infer(resolved);
        require(actual.w == 2 && old[0] == actual[0] && old[1] == actual[1] &&
                actual[0] == 1.f && actual[1] == 5.5f, "Memory and file graph loaders differ");
        rejects([&] { ernie::component_files(dir.path, "head"); });
        rejects([&] { ernie::component_files(dir.path, "../head"); });
        rejects([&] { ernie::load_component(resolved, {}); });
        rejects([&] { ernie::load_component(resolved, {graph + '\0', weights.u8string()}); });
        rejects([&] { ernie::load_component(resolved, {graph, weights.u8string() + '\0'}); });
        { std::ofstream file(param, std::ios::binary); file.put('\0'); }
        rejects([&] { ernie::component_files(dir.path, "head"); });
        { std::ofstream file(param, std::ios::binary); file << std::string(1024 * 1024 + 1, 'x'); }
        rejects([&] { ernie::component_files(dir.path, "head"); });
        std::cout << "Memory/file graph equivalence, lifetime and bounded component inputs passed\n";
    }
    catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

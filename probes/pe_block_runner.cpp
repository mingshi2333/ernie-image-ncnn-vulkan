// SPDX-License-Identifier: MIT
#include "pe_session.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
ncnn::Mat read(const fs::path &path, int width)
{
    ncnn::Mat out(width, 17);
    std::ifstream file(path, std::ios::binary);
    if (fs::file_size(path) != size_t(width) * 17 * 4 ||
        !file.read(static_cast<char *>(out.data), size_t(width) * 17 * 4))
        throw std::runtime_error("Invalid PE fixture tensor");
    return out;
}
int main(int argc, char **argv)
{
    try
    {
        if (argc != 4)
            throw std::invalid_argument("pe-block-runner MODEL FIXTURE NEW_OUTPUT");
        const fs::path model(argv[1]), fixture(argv[2]), output(argv[3]);
        if (fs::exists(output))
            throw std::invalid_argument("Use new PE output");
        const auto input = read(fixture / "x.f32", 3072), cos = read(fixture / "cos.f32", 128),
                   sin = read(fixture / "sin.f32", 128);
        ncnn::Net net;
        net.opt.num_threads = 4;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
        ernie::load_pe_block(net, model.string());
        ernie::PeSession session({&net}, 17), independent({&net}, 17);
        ncnn::Mat actual(3072, 17);
        for (int i = 0; i < 17; ++i)
        {
            auto out = session.step(input.row_range(i, 1), cos.row_range(i, 1), sin.row_range(i, 1));
            std::memcpy(actual.row(i), out.data, 3072 * 4);
            // A separate session starts at a different time; its prefix must
            // remain independent of the longer session sharing the same Net.
            if (i >= 5)
            {
                const int j = i - 5;
                const auto other =
                    independent.step(input.row_range(j, 1), cos.row_range(j, 1), sin.row_range(j, 1));
                if (std::memcmp(other.data, actual.row(j), 3072 * 4))
                    throw std::runtime_error("Independent PE sessions differ");
            }
        }
        const auto changes = session.cache_buffer_changes();
        session.reset();
        for (int i = 0; i < 17; ++i)
        {
            const auto replay = session.step(input.row_range(i, 1), cos.row_range(i, 1), sin.row_range(i, 1));
            if (std::memcmp(replay.data, actual.row(i), 3072 * 4))
                throw std::runtime_error("PE reset/replay differs");
        }
        bool rejected = false;
        try
        {
            session.step(input.row_range(0, 1), cos.row_range(0, 1), sin.row_range(0, 1));
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        if (!rejected || changes != 2 || session.cache_buffer_changes() != 2)
            throw std::runtime_error("PE capacity or cache reuse differs");
        if (output.has_parent_path())
            fs::create_directories(output.parent_path());
        std::ofstream file(output, std::ios::binary);
        file.write(static_cast<const char *>(actual.data), size_t(3072) * 17 * 4);
        if (!file)
            throw std::runtime_error("Cannot write PE result");
        std::cout
            << "17 cached tokens, reset, independent sessions, capacity and two reserved buffers passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

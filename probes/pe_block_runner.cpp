// SPDX-License-Identifier: MIT
#include "pe_session.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
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
        if (argc < 4 || argc > 6)
            throw std::invalid_argument("pe-block-runner MODEL FIXTURE NEW_OUTPUT [PLAN=all1 [THREADS=4]]");
        const fs::path model(argv[1]), fixture(argv[2]), output(argv[3]);
        if (fs::exists(output))
            throw std::invalid_argument("Use new PE output");
        std::vector<int> plan;
        if (argc == 4 || std::string(argv[4]) == "all1")
            plan.assign(17, 1);
        else
        {
            const std::string specification(argv[4]);
            if (specification.empty() || specification.back() == ',')
                throw std::invalid_argument("Invalid PE chunk plan");
            std::istringstream input_plan(specification);
            std::string token;
            while (std::getline(input_plan, token, ','))
            {
                size_t used = 0;
                const int count = std::stoi(token, &used);
                if (used != token.size() || count < 1 || count > 32)
                    throw std::invalid_argument("Invalid PE chunk plan");
                plan.push_back(count);
                if (plan.size() > 17)
                    throw std::invalid_argument("PE chunk plan is too long");
            }
        }
        if (std::accumulate(plan.begin(), plan.end(), 0) != 17)
            throw std::invalid_argument("PE chunk plan must sum to17");
        size_t used = 0;
        const int threads = argc == 6 ? std::stoi(argv[5], &used) : 4;
        if (argc == 6 && used != std::string(argv[5]).size())
            throw std::invalid_argument("Invalid PE threads");
        if (threads < 1 || threads > 256)
            throw std::invalid_argument("Invalid PE threads");
        const auto input = read(fixture / "x.f32", 3072), cos = read(fixture / "cos.f32", 128),
                   sin = read(fixture / "sin.f32", 128);
        const auto saved_input = input.clone(), saved_cos = cos.clone(), saved_sin = sin.clone();
        ncnn::Net net;
        net.opt.num_threads = threads;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
        if (argc >= 5)
            ernie::load_pe_block_chunked(net, model.string());
        else
            ernie::load_pe_block(net, model.string());
        ernie::PeSession session({&net}, 17), independent({&net}, 17);
        auto append = [&](ernie::PeSession &target, int offset, int count)
        {
            return argc >= 5 ? target.append_chunk(input.row_range(offset, count),
                                                   cos.row_range(offset, count), sin.row_range(offset, count))
                             : target.step(input.row_range(offset, count), cos.row_range(offset, count),
                                           sin.row_range(offset, count));
        };
        ncnn::Mat actual(3072, 17);
        int offset = 0, other_offset = 0;
        for (size_t chunk = 0; chunk < plan.size(); ++chunk)
        {
            const int count = plan[chunk];
            const auto out = append(session, offset, count);
            std::memcpy(actual.row(offset), out.data, size_t(3072) * count * 4);
            offset += count;
            // The second session lags one chunk, sharing only immutable weights.
            if (chunk > 0)
            {
                const auto other = append(independent, other_offset, plan[chunk - 1]);
                if (std::memcmp(other.data, actual.row(other_offset), size_t(3072) * plan[chunk - 1] * 4))
                    throw std::runtime_error("Independent PE sessions differ");
                other_offset += plan[chunk - 1];
            }
        }
        const auto other = append(independent, other_offset, plan.back());
        if (std::memcmp(other.data, actual.row(other_offset), size_t(3072) * plan.back() * 4))
            throw std::runtime_error("Independent PE sessions differ");
        const auto changes = session.cache_buffer_changes();
        session.reset();
        offset = 0;
        for (int count : plan)
        {
            const auto replay = append(session, offset, count);
            if (std::memcmp(replay.data, actual.row(offset), size_t(3072) * count * 4))
                throw std::runtime_error("PE reset/replay differs");
            offset += count;
        }
        if (std::memcmp(input.data, saved_input.data, size_t(3072) * 17 * 4) ||
            std::memcmp(cos.data, saved_cos.data, size_t(128) * 17 * 4) ||
            std::memcmp(sin.data, saved_sin.data, size_t(128) * 17 * 4))
            throw std::runtime_error("PE mutated fixture inputs");
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

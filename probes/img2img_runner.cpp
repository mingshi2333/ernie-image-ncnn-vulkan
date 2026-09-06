// SPDX-License-Identifier: MIT
#include "image_encoder.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

std::string read_text(const fs::path &path)
{
    const auto size = fs::file_size(path);
    if (!size || size > 1024 * 1024) throw std::invalid_argument("Invalid encoder graph size");
    std::string result(size, '\0'); std::ifstream stream(path, std::ios::binary);
    if (!stream.read(result.data(), std::streamsize(size)) || result.find('\0') != std::string::npos)
        throw std::runtime_error("Cannot read encoder graph");
    return result;
}

void save(const fs::path &path, const ncnn::Mat &value)
{
    std::ofstream stream(path, std::ios::binary | std::ios::out);
    for (int c = 0; c < value.c; ++c)
        if (!stream.write(reinterpret_cast<const char *>(static_cast<const float *>(value.channel(c))),
                          std::streamsize(value.w) * value.h * sizeof(float)))
            throw std::runtime_error("Cannot write encoder boundary");
    stream.close(); if (!stream) throw std::runtime_error("Cannot close encoder boundary");
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 7) throw std::invalid_argument("Usage: runner PARAM BIN INPUT.RGB OUTPUT WIDTH HEIGHT");
        const fs::path param = argv[1], weights = argv[2], input = argv[3], output = argv[4];
        const int width = std::stoi(argv[5]), height = std::stoi(argv[6]);
        if (fs::exists(output)) throw std::invalid_argument("Use a new output directory");
        const auto count = size_t(width) * size_t(height) * 3;
        if (fs::file_size(input) != count) throw std::invalid_argument("RGB byte count differs");
        ernie::RgbImage rgb{width, height, std::vector<uint8_t>(count)};
        std::ifstream source(input, std::ios::binary);
        if (!source.read(reinterpret_cast<char *>(rgb.pixels.data()), std::streamsize(count)))
            throw std::runtime_error("Cannot read RGB input");
        ncnn::Option cpu; cpu.num_threads = 2; cpu.use_packing_layout = false;
        cpu.use_winograd_convolution = false; cpu.use_sgemm_convolution = false;
        cpu.use_fp16_storage = cpu.use_fp16_arithmetic = cpu.use_fp16_packed = false;
        cpu.use_bf16_storage = cpu.use_bf16_packed = false;
        const auto result = ernie::encode_vae({read_text(param), weights.string()}, rgb, cpu);
        fs::create_directories(output);
        save(output / "mean.f32", result.mean);
        save(output / "packed.f32", result.packed);
        save(output / "normalized.f32", result.normalized);
        std::cout << "native VAE encoder boundaries saved\n";
    }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

// SPDX-License-Identifier: MIT
#include "image_encoder.h"
#include "img2img.h"
#include "latent_ops.h"
#include "vae.h"
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

ncnn::Mat read_f32(const fs::path &path, int width, int height, int channels)
{
    ncnn::Mat result(width, height, channels);
    if (fs::file_size(path) != result.total() * sizeof(float)) throw std::invalid_argument("FP32 input size differs");
    std::ifstream stream(path, std::ios::binary);
    for (int c = 0; c < channels; ++c)
        if (!stream.read(reinterpret_cast<char *>(static_cast<float *>(result.channel(c))),
                         std::streamsize(width) * height * sizeof(float)))
            throw std::runtime_error("Cannot read FP32 input");
    return result;
}

ncnn::Mat read_vector(const fs::path &path, int count)
{
    ncnn::Mat result(count);
    if (fs::file_size(path) != size_t(count) * sizeof(float)) throw std::invalid_argument("FP32 vector size differs");
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(static_cast<float *>(result)), std::streamsize(count) * sizeof(float)))
        throw std::runtime_error("Cannot read FP32 vector");
    return result;
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 10) throw std::invalid_argument("Usage: runner ENCODER.PARAM ENCODER.BIN INPUT.RGB OUTPUT WIDTH HEIGHT BN-MEAN BN-VAR DECODER-DIR");
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
        const auto schedule = ernie::FlowSchedule::turbo(8);
        const auto start = ernie::make_img2img_start(result.normalized, ncnn::Mat(), schedule, 0.f, 2);
        const auto mean = read_vector(argv[7], 128);
        const auto variance = read_vector(argv[8], 128);
        const auto unpacked = ernie::unpack_for_vae(start.latent, mean, variance, 2);
        const auto decoded = ernie::decode_vae(argv[9], unpacked, cpu, "cpu", "direct", -1);
        fs::create_directories(output);
        save(output / "mean.f32", result.mean);
        save(output / "packed.f32", result.packed);
        save(output / "normalized.f32", result.normalized);
        save(output / "start.f32", start.latent);
        save(output / "unpacked.f32", unpacked);
        save(output / "decoded.f32", decoded);
        std::cout << "native VAE encoder boundaries saved\n";
    }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

// SPDX-License-Identifier: MIT
#pragma once
#include <ernie/pipeline.h>
#include <array>
#include <filesystem>
namespace ernie::cli
{
RgbImage read_image(const std::filesystem::path &path,
                    const std::array<uint8_t, 3> &background = {255, 255, 255});
void write_image(const std::filesystem::path &path, const RgbImage &image);
void write_png(const std::filesystem::path &path, const RgbImage &image);
}

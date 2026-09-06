// SPDX-License-Identifier: MIT
#pragma once
#include <ernie/pipeline.h>
#include <filesystem>
namespace ernie::cli
{
void write_png(const std::filesystem::path &path, const RgbImage &image);
}

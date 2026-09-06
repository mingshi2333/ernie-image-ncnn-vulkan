// SPDX-License-Identifier: MIT
#pragma once
#include "mat.h"
#include <filesystem>
#include <string>
namespace ernie
{
ncnn::Mat read_tensor(const std::filesystem::path &path, int w, int h = 1, int c = 1);
void write_tensor(const std::filesystem::path &path, const ncnn::Mat &value);
std::string numbered(const std::string &stem, int index);
} // namespace ernie

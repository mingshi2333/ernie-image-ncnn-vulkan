// SPDX-License-Identifier: MIT
#pragma once
#include <filesystem>
#include <string>
namespace ernie::cli
{
std::string read_prompt(const std::filesystem::path &path);
}

// SPDX-License-Identifier: MIT
#pragma once
#include <filesystem>
namespace ernie
{
struct ModelConfig
{
    int packed_width, packed_height, text_bucket, dit_text_tokens;
};
ModelConfig model_config(const std::filesystem::path &path);
} // namespace ernie

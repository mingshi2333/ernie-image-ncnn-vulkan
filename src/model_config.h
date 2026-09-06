// SPDX-License-Identifier: MIT
#pragma once
#include <filesystem>
namespace ernie
{
struct ModelConfig
{
    int packed_width, packed_height, text_bucket, dit_text_tokens;
};
void validate_model_config(const ModelConfig &config);
// Restricts offline graph candidates to the three existing audited static instances.
bool reviewed_shape_config(const ModelConfig &config);
ModelConfig model_config(const std::filesystem::path &path);
} // namespace ernie

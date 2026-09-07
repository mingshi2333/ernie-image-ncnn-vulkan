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
// Schema-3 spatial planning has separate limits; legacy/static exports retain
// their original 6144-token protection.
void validate_runtime_model_config(const ModelConfig &config);
// Authenticated source templates; runtime target bounds are checked separately.
bool reviewed_shape_config(const ModelConfig &config);
ModelConfig model_config(const std::filesystem::path &path, bool runtime_shape = false);
} // namespace ernie

// SPDX-License-Identifier: MIT
#include "model_config.h"
#include "shape_plan.h"
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
namespace fs = std::filesystem;
namespace ernie
{
void validate_model_config(const ModelConfig &c)
{
    if (c.packed_width < 1 || c.packed_width > 128 || c.packed_height < 1 || c.packed_height > 128 ||
        c.text_bucket < 1 || c.text_bucket > 2048 || c.dit_text_tokens < c.text_bucket ||
        c.dit_text_tokens > 2048 || c.packed_width * c.packed_height + c.dit_text_tokens > 6144)
        throw std::runtime_error("Unsupported model configuration");
}
bool reviewed_shape_config(const ModelConfig &c)
{
    return (c.packed_width == 4 && c.packed_height == 4 && c.text_bucket == 32 && c.dit_text_tokens == 272) ||
           (c.packed_width == 32 && c.packed_height == 24 && c.text_bucket == 2048 && c.dit_text_tokens == 2048) ||
           (c.packed_width == 64 && c.packed_height == 64 && c.text_bucket == 64 && c.dit_text_tokens == 64) ||
           (c.packed_width == 64 && c.packed_height == 64 && c.text_bucket == 32 && c.dit_text_tokens == 64) ||
           (c.packed_width == 86 && c.packed_height == 48 && c.text_bucket == 64 && c.dit_text_tokens == 64);
}
void validate_runtime_model_config(const ModelConfig &c)
{
    if (c.packed_width < 1 || c.packed_width > 128 || c.packed_height < 1 || c.packed_height > 128 ||
        c.dit_text_tokens < c.text_bucket)
        throw std::invalid_argument("Unsupported runtime model configuration");
    ShapePlan::create({{c.text_bucket}, c.dit_text_tokens},
                      c.packed_width * 16, c.packed_height * 16, c.text_bucket);
}
ModelConfig model_config(const fs::path &path, bool runtime_shape)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open model.cfg");
    std::map<std::string, int> values;
    std::string key, raw;
    while (file >> key)
    {
        if (!(file >> raw) || values.count(key))
            throw std::runtime_error("Malformed or duplicate model.cfg entry");
        size_t used = 0;
        const int value = std::stoi(raw, &used);
        if (used != raw.size())
            throw std::runtime_error("Invalid model.cfg integer");
        values[key] = value;
    }
    for (const auto *name :
         {"packed_width", "packed_height", "text_bucket", "dit_text_tokens", "text_layers", "dit_layers"})
        if (!values.count(name))
            throw std::runtime_error("Missing model.cfg field");
    const int w = values.at("packed_width"), h = values.at("packed_height"), t = values.at("dit_text_tokens"),
              bucket = values.at("text_bucket");
    if (values.size() != 6 || w < 1 || h < 1 || w > 128 || h > 128 || bucket < 1 || bucket > 2048 ||
        t < bucket || t > 2048 || (!runtime_shape && w * h + t > 6144) || values.at("text_layers") != 25 ||
        values.at("dit_layers") != 36)
        throw std::runtime_error("Unsupported model configuration");
    ModelConfig c{w, h, bucket, t};
    if (runtime_shape) validate_runtime_model_config(c);
    return c;
}
} // namespace ernie

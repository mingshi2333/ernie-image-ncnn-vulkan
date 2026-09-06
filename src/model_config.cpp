// SPDX-License-Identifier: MIT
#include "model_config.h"
#include <fstream>
#include <map>
#include <stdexcept>
namespace fs = std::filesystem;
namespace ernie
{
ModelConfig model_config(const fs::path &path)
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
        t < bucket || t > 2048 || w * h + t > 6144 || values.at("text_layers") != 25 ||
        values.at("dit_layers") != 36)
        throw std::runtime_error("Unsupported model configuration");
    return {w, h, bucket, t};
}
} // namespace ernie

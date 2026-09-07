// SPDX-License-Identifier: MIT
#pragma once
#include "mat.h"
#include <ernie/pipeline.h>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace ernie
{
struct PeResult
{
    std::string text;
    std::vector<uint32_t> input_ids, generated_ids;
    bool eos = false;
    size_t cache_buffer_changes = 0;
};
uint32_t sample_pe_token(const ncnn::Mat &logits, const PeOptions &options, std::mt19937 &random);
using PeProgress = std::function<void(const char *, int, int)>;
using PeLogits = std::function<void(int, const ncnn::Mat &)>;
// Loads the optional CPU model, enhances one prompt, then releases all weights
// and caches before the image text encoder/DiT begin.
// Internal candidate only: chunk1 retains the legacy graph/sequential prefill.
PeResult enhance_prompt(const std::string &model, const std::string &prompt, int width, int height,
                        const PeOptions &options, const PeProgress &progress = {},
                        const PeLogits &logits = {}, int threads = 4, int prefill_chunk = 1);
} // namespace ernie

// SPDX-License-Identifier: MIT
#pragma once
#include "mat.h"
#include <string>
#include <vector>
namespace ernie
{
// Official 3-axis frequency table, 64 FP32 values (16 + 24 + 24).
// Output order: cosine, sine, additive attention mask. Adjacent duplication
// implements ERNIE's full-width angles, not the Llama half-table convention.
std::vector<ncnn::Mat> dit_constants(const std::string &frequency_path, int width, int height, int valid_text,
                                     int text_bucket);
ncnn::Mat pad_text(const ncnn::Mat &embeddings, int bucket);
} // namespace ernie

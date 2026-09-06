// SPDX-License-Identifier: MIT
#pragma once
#include "block_sequence.h"
#include <cstdint>

namespace ernie
{
enum class TextDownMode { Gemm, Vector };

// The converted embedding table is raw BF16, vocabulary=131072, width=3072.
// Only requested rows are read. Padding rows cannot influence earlier causal tokens.
ncnn::Mat text_embeddings(const std::string &path, const std::vector<uint32_t> &ids, int bucket);
// The converter stores the exact official YaRN inverse frequencies; the
// configured attention factor is 1. Text prefill does not use an AR K/V cache.
std::vector<ncnn::Mat> text_constants(const std::string &inverse_frequency_path, int bucket);
ncnn::Mat run_text_blocks(const std::vector<std::string> &models, const ncnn::Mat &input,
                          const std::vector<ncnn::Mat> &constants, const ncnn::Option &option,
                          BlockSequenceStats &stats, TextDownMode down_mode = TextDownMode::Gemm);
#if NCNN_VULKAN
ncnn::VkMat run_text_blocks(const std::vector<std::string> &models, const ncnn::VkMat &input,
                            const std::vector<ncnn::VkMat> &constants, const ncnn::VulkanDevice *device,
                            const ncnn::Option &option, BlockSequenceStats &stats);
#endif
} // namespace ernie

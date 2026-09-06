// SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace ernie
{
// Mathematical limits only. This is not a validated graph or generation contract.
struct ShapeContract
{
    std::array<int, 3> text_buckets{32, 64, 2048};
};
std::size_t checked_shape_product(std::size_t a, std::size_t b);
std::size_t checked_shape_sum(std::size_t a, std::size_t b);
struct ShapePlan
{
    int width, height, latent_width, latent_height, packed_width, packed_height;
    int valid_text_tokens, text_bucket;
    std::size_t image_tokens, total_tokens, valid_tokens;
    std::size_t packed_latent_elements, packed_latent_bytes, rgb_bytes;
    std::size_t rope_elements, mask_elements;
    // No graph instantiation, RoPE values, padding-mask values or inference is performed.
    static ShapePlan create(const ShapeContract &, std::int64_t width, std::int64_t height,
                            std::int64_t valid_text_tokens);
};
} // namespace ernie

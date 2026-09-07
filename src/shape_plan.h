// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace ernie
{
// Available independently exported text buckets. Spatial graph authentication
// remains in ModelPackage; this planner never accepts arbitrary graph rewrites.
struct ShapeContract
{
    std::vector<int> text_buckets{32, 64, 2048};
    int minimum_dit_text_tokens = 0;
};
std::size_t checked_shape_product(std::size_t a, std::size_t b);
std::size_t checked_shape_sum(std::size_t a, std::size_t b);
struct ShapePlan
{
    int width, height, latent_width, latent_height, packed_width, packed_height;
    int valid_text_tokens, text_bucket, dit_text_tokens;
    std::size_t image_tokens, total_tokens, valid_tokens;
    std::size_t packed_latent_elements, packed_latent_bytes, rgb_bytes;
    std::size_t rope_elements, mask_elements;
    // Larger sequences use host-resident Vulkan weights based on the real-weight
    // 10240-token block experiment. This is a policy, not a memory-use guarantee.
    bool host_weights;
    static ShapePlan create(const ShapeContract &, std::int64_t width, std::int64_t height,
                            std::int64_t valid_text_tokens);
};
} // namespace ernie

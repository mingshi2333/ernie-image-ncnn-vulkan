// SPDX-License-Identifier: MIT
#include "shape_plan.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace ernie
{
std::size_t checked_shape_product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Shape product overflow");
    return a * b;
}
std::size_t checked_shape_sum(std::size_t a, std::size_t b)
{
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error("Shape sum overflow");
    return a + b;
}
ShapePlan ShapePlan::create(const ShapeContract &contract, std::int64_t w, std::int64_t h,
                            std::int64_t tokens)
{
    int previous = 0;
    for (int bucket : contract.text_buckets)
    {
        if ((bucket != 32 && bucket != 64 && bucket != 2048) || bucket <= previous)
            throw std::invalid_argument("Text buckets must be an ordered subset of 32/64/2048");
        previous = bucket;
    }
    if (contract.text_buckets.empty() || contract.minimum_dit_text_tokens < 0 ||
        contract.minimum_dit_text_tokens > 2048)
        throw std::invalid_argument("Invalid text padding contract");
    if (w < 16 || w > 2048 || h < 16 || h > 2048 || w % 16 || h % 16 || tokens < 1 || tokens > 2048)
        throw std::invalid_argument("Unsupported mathematical shape");
    const auto area = checked_shape_product(static_cast<std::size_t>(w), static_cast<std::size_t>(h));
    if (area > 2097152)
        throw std::invalid_argument("Mathematical image area limit exceeded");
    ShapePlan s{};
    s.width = static_cast<int>(w); s.height = static_cast<int>(h);
    s.latent_width = s.width / 8; s.latent_height = s.height / 8;
    s.packed_width = s.width / 16; s.packed_height = s.height / 16;
    s.valid_text_tokens = static_cast<int>(tokens);
    for (int bucket : contract.text_buckets)
        if (tokens <= bucket) { s.text_bucket = bucket; break; }
    if (!s.text_bucket)
        throw std::invalid_argument("Prompt exceeds the available text buckets");
    s.dit_text_tokens = std::max(s.text_bucket, contract.minimum_dit_text_tokens);
    s.image_tokens = checked_shape_product(s.packed_width, s.packed_height);
    s.total_tokens = checked_shape_sum(s.image_tokens, s.dit_text_tokens);
    s.valid_tokens = checked_shape_sum(s.image_tokens, s.valid_text_tokens);
    s.packed_latent_elements = checked_shape_product(128, s.image_tokens);
    s.packed_latent_bytes = checked_shape_product(4, s.packed_latent_elements);
    s.rgb_bytes = checked_shape_product(3, area);
    s.rope_elements = checked_shape_product(s.total_tokens, 128); // each of cosine and sine
    s.mask_elements = checked_shape_product(s.total_tokens, s.total_tokens); // current dense additive mask
    s.host_weights = s.total_tokens > 6144;
    return s;
}
} // namespace ernie

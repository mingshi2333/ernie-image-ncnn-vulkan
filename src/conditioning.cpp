// SPDX-License-Identifier: MIT
#include "conditioning.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
namespace ernie
{
static std::vector<ncnn::Mat> make_constants(const std::string &path, int w, int h, int valid, int text,
                                            int token_limit)
{
    if (w < 1 || w > 256 || h < 1 || h > 256 || valid < 1 || text < valid || text > 2048 ||
        w * h + text > token_limit || std::filesystem::file_size(path) != 64 * 4)
        throw std::invalid_argument("Invalid DiT position dimensions or frequency file");
    float frequencies[64];
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(frequencies), sizeof(frequencies)))
        throw std::runtime_error("Cannot read DiT frequencies");
    for (float f : frequencies)
        if (!std::isfinite(f) || f <= 0.f)
            throw std::runtime_error("Invalid DiT frequency");
    const int image = w * h, length = image + text;
    ncnn::Mat cos(128, length), sin(128, length), mask(length, length);
    if (cos.empty() || sin.empty() || mask.empty())
        throw std::bad_alloc();
    for (int i = 0; i < length; ++i)
    {
        const int positions[3] = {i < image ? valid : i - image, i < image ? i / w : 0,
                                  i < image ? i % w : 0};
        float *c = cos.row(i);
        float *s = sin.row(i);
        float *m = mask.row(i);
        for (int j = 0; j < 64; ++j)
        {
            const int axis = j < 16 ? 0 : j < 40 ? 1 : 2;
            const float phase = float(positions[axis]) * frequencies[j];
            c[2 * j] = c[2 * j + 1] = std::cos(phase);
            s[2 * j] = s[2 * j + 1] = std::sin(phase);
        }
        for (int k = 0; k < length; ++k)
            m[k] = k >= image + valid ? -1e30f : 0.f;
    }
    return {cos, sin, mask};
}
std::vector<ncnn::Mat> dit_constants(const std::string &path, int w, int h, int valid, int text)
{
    return make_constants(path, w, h, valid, text, 6144);
}
std::vector<ncnn::Mat> dit_constants(const std::string &path, const ShapePlan &shape)
{
    const auto checked = ShapePlan::create({{shape.text_bucket}, shape.dit_text_tokens},
                                            shape.width, shape.height, shape.valid_text_tokens);
    return make_constants(path, checked.packed_width, checked.packed_height,
                          checked.valid_text_tokens, checked.dit_text_tokens, 10240);
}
ncnn::Mat pad_text(const ncnn::Mat &embeddings, int bucket)
{
    if (embeddings.empty() || embeddings.dims != 2 || embeddings.w != 3072 || embeddings.h < 1 ||
        embeddings.h > bucket || bucket > 2048 || embeddings.elempack != 1 || embeddings.elemsize != 4u)
        throw std::invalid_argument("Invalid text embeddings or padding bucket");
    ncnn::Mat out(3072, bucket);
    if (out.empty())
        throw std::bad_alloc();
    out.fill(0.f);
    std::memcpy(out.data, embeddings.data, size_t(3072) * embeddings.h * 4);
    return out;
}
} // namespace ernie

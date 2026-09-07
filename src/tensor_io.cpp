// SPDX-License-Identifier: MIT
#include "tensor_io.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
namespace fs = std::filesystem;
namespace ernie
{
ncnn::Mat read_tensor(const fs::path &path, int w, int h, int c)
{
    if (fs::file_size(path) != size_t(w) * h * c * 4)
        throw std::runtime_error("Wrong tensor size: " + path.u8string());
    ncnn::Mat out = c > 1 ? ncnn::Mat(w, h, c) : h > 1 ? ncnn::Mat(w, h) : ncnn::Mat(w);
    if (out.empty())
        throw std::bad_alloc();
    std::ifstream file(path, std::ios::binary);
    for (int k = 0; k < c; ++k)
    {
        if (!file.read(static_cast<char *>(out.channel(k).data), size_t(w) * h * 4))
            throw std::runtime_error("Tensor read failed");
        const float *values = out.channel(k);
        if (!std::all_of(values, values + size_t(w) * h, [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Tensor contains non-finite values: " + path.u8string());
    }
    return out;
}
void write_tensor(const fs::path &path, const ncnn::Mat &value)
{
    if (fs::exists(path))
        throw std::runtime_error("Trace file exists");
    if (value.empty() || value.elempack != 1 || value.elemsize != 4u)
        throw std::runtime_error("Trace tensor must be FP32");
    std::ofstream out(path, std::ios::binary);
    for (int c = 0; c < value.c; ++c)
        out.write(static_cast<const char *>(value.channel(c).data), size_t(value.w) * value.h * value.d * 4);
    if (!out)
        throw std::runtime_error("Trace write failed");
}
std::string numbered(const std::string &stem, int i)
{
    return stem + (i < 10 ? "0" : "") + std::to_string(i);
}
} // namespace ernie

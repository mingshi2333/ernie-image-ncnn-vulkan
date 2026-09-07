// SPDX-License-Identifier: MIT
#include "pe_session.h"
#include "ernie_gelu.h"
#include "pe_graph.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ernie
{
namespace
{
void check(int result, const char *action)
{
    if (result)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(result));
}
bool fp32_cpu(const ncnn::Option &opt)
{
    return !opt.use_vulkan_compute && !opt.use_fp16_storage && !opt.use_fp16_packed &&
           !opt.use_fp16_arithmetic && !opt.use_bf16_storage && !opt.use_bf16_packed;
}
void matrix_check(const ncnn::Mat &value, int width, int rows, bool legacy = false)
{
    if (value.empty() || value.elempack != 1 || value.elemsize != 4u || value.w != width || value.h != rows ||
        value.c != 1 || value.d != 1 ||
        (value.dims != 2 && !(legacy && rows == 1 && (value.dims == 1 || value.dims == 3))))
        throw std::invalid_argument("PE requires pack1 FP32 matrices with matching token rows");
    const float *p = value;
    for (int i = 0; i < width * rows; ++i)
        if (!std::isfinite(p[i]))
            throw std::invalid_argument("PE tensor contains non-finite values");
}
} // namespace

void load_pe_block(ncnn::Net &net, const std::string &directory)
{
    if (!fp32_cpu(net.opt))
        throw std::invalid_argument("PE currently requires CPU FP32");
    const auto root = std::filesystem::u8path(directory);
    check(register_layers(net), "Register PE layers");
    check(net.load_param((root / "pe.ncnn.param").c_str()), "Load PE graph");
    check(net.load_model((root / "pe.ncnn.bin").c_str()), "Load PE weights");
}

void load_pe_block_chunked(ncnn::Net &net, const std::string &directory)
{
    if (!fp32_cpu(net.opt))
        throw std::invalid_argument("PE currently requires CPU FP32");
    const auto root = std::filesystem::u8path(directory);
    const auto path = root / "pe.ncnn.param";
    if (std::filesystem::file_size(path) > 16384)
        throw std::invalid_argument("Unknown PE graph");
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot read PE graph");
    const std::string bytes((std::istreambuf_iterator<char>(file)), {});
    const auto graph = pe_chunk_graph(bytes); // Authenticate before loading any weights.
    check(register_layers(net), "Register PE layers");
    check(net.load_param_mem(graph.c_str()), "Load chunked PE graph");
    check(net.load_model((root / "pe.ncnn.bin").c_str()), "Load PE weights");
}

PeSession::PeSession(std::vector<const ncnn::Net *> blocks, int capacity)
    : blocks_(std::move(blocks)), keys_(blocks_.size()), values_(blocks_.size()), capacity_(capacity)
{
    if (blocks_.empty() || blocks_.size() > 26 || capacity < 1 || capacity > 4096)
        throw std::invalid_argument("PE requires 1..26 blocks and capacity 1..4096");
    for (const auto *block : blocks_)
        if (!block || !fp32_cpu(block->opt))
            throw std::invalid_argument("PE session requires CPU FP32 models");
    cache_allocator_.set_size_compare_ratio(0.f);
}

void PeSession::reset()
{
    for (auto &key : keys_)
        key.release();
    for (auto &value : values_)
        value.release();
    position_ = 0;
    buffer_changes_ = 0;
    valid_ = true;
}

ncnn::Mat PeSession::step(const ncnn::Mat &embedded, const ncnn::Mat &cos, const ncnn::Mat &sin)
{
    return append_checked(embedded, cos, sin, true);
}

ncnn::Mat PeSession::append_chunk(const ncnn::Mat &embedded, const ncnn::Mat &cos, const ncnn::Mat &sin)
{
    return append_checked(embedded, cos, sin, false);
}

ncnn::Mat PeSession::append_checked(const ncnn::Mat &embedded, const ncnn::Mat &cos, const ncnn::Mat &sin,
                                    bool single)
{
    if (!valid_)
        throw std::runtime_error("PE session failed; reset before reuse");
    const int rows = embedded.h;
    if (rows < 1 || rows > (single ? 1 : 32) || rows > capacity_ - position_)
        throw std::invalid_argument("PE token count or session capacity exceeded");
    matrix_check(embedded, 3072, rows, single);
    matrix_check(cos, 128, rows, single);
    matrix_check(sin, 128, rows, single);
    ncnn::Mat mask(position_ + rows, rows);
    if (mask.empty())
        throw std::bad_alloc();
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < mask.w; ++j)
            mask.row(r)[j] = j <= position_ + r ? 0.f : std::numeric_limits<float>::lowest();
    // All validation/allocation before cache mutation preserves a valid session.
    // Clone before reshape: external/row views cannot enter in-place layers.
    ncnn::Mat current = embedded.clone().reshape(3072, rows);
    const auto owned_cos = cos.clone().reshape(128, rows), owned_sin = sin.clone().reshape(128, rows);
    if (current.empty() || owned_cos.empty() || owned_sin.empty())
        throw std::bad_alloc();
    try
    {
        for (size_t i = 0; i < blocks_.size(); ++i)
        {
            auto ex = blocks_[i]->create_extractor();
            ex.set_kvcache_allocator(&cache_allocator_);
            ex.set_kvcache_max_seqlen_hint(capacity_);
            check(ex.input("in0", current), "Input PE hidden state");
            check(ex.input("in1", owned_cos), "Input PE cosine");
            check(ex.input("in2", owned_sin), "Input PE sine");
            check(ex.input("in3", mask), "Input PE causal mask");
            const void *old_key = keys_[i].data, *old_value = values_[i].data;
            if (!keys_[i].empty())
            {
                check(ex.input("past_k", keys_[i]), "Input PE key cache");
                check(ex.input("past_v", values_[i]), "Input PE value cache");
                keys_[i].release();
                values_[i].release();
            }
            // type=1 preserves the opaque native cache and its reserved capacity.
            check(ex.extract("out_k", keys_[i], 1), "Extract PE key cache");
            check(ex.extract("out_v", values_[i], 1), "Extract PE value cache");
            ncnn::Mat next;
            check(ex.extract("out0", next), "Extract PE hidden state");
            if (keys_[i].empty() || values_[i].empty() || keys_[i].h != position_ + rows ||
                values_[i].h != position_ + rows || keys_[i].allocator != &cache_allocator_ ||
                values_[i].allocator != &cache_allocator_)
                throw std::runtime_error("PE cache length, presence or session allocator differs");
            buffer_changes_ += (old_key != keys_[i].data) + (old_value != values_[i].data);
            matrix_check(next, 3072, rows);
            if (!next.refcount)
                throw std::runtime_error("PE hidden output is not owned");
            current = next;
        }
        position_ += rows;
        return current;
    }
    catch (...)
    {
        reset();
        valid_ = false;
        throw;
    }
}
} // namespace ernie

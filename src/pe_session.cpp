// SPDX-License-Identifier: MIT
#include "pe_session.h"
#include "ernie_gelu.h"
#include <cmath>
#include <filesystem>
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
void vector_check(const ncnn::Mat &value, int width)
{
    if (value.empty() || value.elempack != 1 || value.elemsize != 4u || value.w != width || value.h != 1 ||
        value.c != 1 || value.d != 1)
        throw std::invalid_argument("PE requires one FP32 token per call");
    const float *p = value;
    for (int i = 0; i < width; ++i)
        if (!std::isfinite(p[i]))
            throw std::invalid_argument("PE tensor contains non-finite values");
}
} // namespace

void load_pe_block(ncnn::Net &net, const std::string &directory)
{
    if (!fp32_cpu(net.opt))
        throw std::invalid_argument("PE currently requires CPU FP32");
    const std::filesystem::path root(directory);
    check(register_layers(net), "Register PE layers");
    check(net.load_param((root / "pe.ncnn.param").string().c_str()), "Load PE graph");
    check(net.load_model((root / "pe.ncnn.bin").string().c_str()), "Load PE weights");
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
    if (!valid_)
        throw std::runtime_error("PE session failed; reset before reuse");
    if (position_ >= capacity_)
        throw std::invalid_argument("PE session capacity exhausted");
    vector_check(embedded, 3072);
    vector_check(cos, 128);
    vector_check(sin, 128);
    ncnn::Mat mask(position_ + 1, 1);
    if (mask.empty())
        throw std::bad_alloc();
    mask.fill(0.f); // Only past and current tokens exist in this cache.
    // row_range/external Mat views have no owning refcount. ncnn in-place
    // layers require owned inputs when lightmode is enabled.
    ncnn::Mat current = embedded.clone();
    const auto owned_cos = cos.clone(), owned_sin = sin.clone();
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
            if (keys_[i].allocator != &cache_allocator_ || values_[i].allocator != &cache_allocator_)
                throw std::runtime_error("PE cache escaped its session allocator");
            buffer_changes_ += (old_key != keys_[i].data) + (old_value != values_[i].data);
            vector_check(next, 3072);
            current = next.reshape(3072, 1);
        }
        ++position_;
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

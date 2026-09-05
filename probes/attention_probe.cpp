// SPDX-License-Identifier: MIT
#include "net.h"
#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int q_heads = 32;
constexpr int kv_heads = 8;
constexpr int head_dim = 128;
constexpr char graph[] =
    "7767517\n6 9\n"
    "Input q_input 0 1 q\n"
    "Input k_input 0 1 k\n"
    "Input v_input 0 1 v\n"
    "Input mask_input 0 1 mask\n"
    "Input cache_input 0 2 past_k past_v\n"
    "SDPA attention 6 3 q k v mask past_k past_v out out_k out_v 5=1 7=1\n";
constexpr unsigned int empty_model[] = {0};

void check(int result, const char* operation)
{
    if (result != 0)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}

float sample(int head, int pos, int dim, unsigned seed)
{
    uint32_t x = seed ^ (uint32_t(head + 1) * 0x9e3779b9u)
                      ^ (uint32_t(pos + 1) * 0x85ebca6bu)
                      ^ (uint32_t(dim + 1) * 0xc2b2ae35u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return (static_cast<int>(x & 65535u) - 32768) / 32768.f;
}

ncnn::Mat tensor(int heads, int len, int start, unsigned seed)
{
    ncnn::Mat result(head_dim, len, heads);
    for (int h = 0; h < heads; ++h)
    {
        ncnn::Mat channel = result.channel(h);
        for (int p = 0; p < len; ++p)
            for (int d = 0; d < head_dim; ++d)
                channel.row(p)[d] = sample(h, start + p, d, seed);
    }
    return result;
}

struct Inputs {
    ncnn::Mat q, k, v, mask;
};

Inputs inputs(int len, int start, unsigned seed)
{
    Inputs in{tensor(q_heads, len, start, seed), tensor(kv_heads, len, start, seed + 1),
              tensor(kv_heads, len, start, seed + 2), ncnn::Mat(start + len, len)};
    for (int p = 0; p < len; ++p)
        for (int k = 0; k < start + len; ++k)
            in.mask.row(p)[k] = k <= start + p ? 0.f : -1e30f;
    return in;
}

// An independent, double-accumulating causal GQA oracle. It never reads ncnn's cache.
ncnn::Mat reference(const Inputs& in, int start, unsigned seed)
{
    const int len = in.q.h;
    const ncnn::Mat keys = tensor(kv_heads, start + len, 0, seed + 1);
    const ncnn::Mat values = tensor(kv_heads, start + len, 0, seed + 2);
    ncnn::Mat out(head_dim, len, q_heads);
    for (int h = 0; h < q_heads; ++h)
    {
        const ncnn::Mat q = in.q.channel(h);
        const ncnn::Mat k = keys.channel(h / (q_heads / kv_heads));
        const ncnn::Mat v = values.channel(h / (q_heads / kv_heads));
        ncnn::Mat dst = out.channel(h);
        for (int p = 0; p < len; ++p)
        {
            const int visible = start + p + 1;
            std::vector<double> scores(visible);
            double maximum = -1e100;
            for (int j = 0; j < visible; ++j)
            {
                double score = 0;
                for (int d = 0; d < head_dim; ++d)
                    score += double(q.row(p)[d]) * k.row(j)[d];
                scores[j] = score / std::sqrt(double(head_dim));
                maximum = std::max(maximum, scores[j]);
            }
            double sum = 0;
            for (double& score : scores)
            {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (int d = 0; d < head_dim; ++d)
            {
                double value = 0;
                for (int j = 0; j < visible; ++j)
                    value += scores[j] * v.row(j)[d];
                dst.row(p)[d] = float(value / sum);
            }
        }
    }
    return out;
}

void configure(ncnn::Option& opt, const std::string& precision)
{
    opt.num_threads = 4;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = precision == "fp16";
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = precision == "bf16";
}

void feed(ncnn::Extractor& ex, const Inputs& in)
{
    check(ex.input("q", in.q), "input q");
    check(ex.input("k", in.k), "input k");
    check(ex.input("v", in.v), "input v");
    check(ex.input("mask", in.mask), "input mask");
}

struct Session {
    virtual ~Session() = default;
    virtual ncnn::Mat step(const Inputs&, int hint) = 0;
    virtual void reset() = 0;
    // Count changes of the backing allocation, not allocator calls or cache bytes.
    int key_buffers = 0;
    int value_buffers = 0;
};

struct CpuSession final : Session {
    // Cache storage must be destroyed before its session allocator.
    ncnn::UnlockedPoolAllocator allocator;
    ncnn::Net net;
    ncnn::Mat key, value;
    bool dedicated;

    explicit CpuSession(bool use_allocator) : dedicated(use_allocator)
    {
        allocator.set_size_compare_ratio(0.f);
        configure(net.opt, "fp32");
        net.opt.use_vulkan_compute = false;
        check(net.load_param_mem(graph), "load CPU graph");
        check(net.load_model(reinterpret_cast<const unsigned char*>(empty_model)), "load CPU model");
    }

    ncnn::Mat step(const Inputs& in, int hint) override
    {
        const void* old_key = key.data;
        const void* old_value = value.data;
        ncnn::Extractor ex = net.create_extractor();
        if (dedicated)
            ex.set_kvcache_allocator(&allocator);
        ex.set_kvcache_max_seqlen_hint(hint);
        feed(ex, in);
        if (!key.empty())
        {
            check(ex.input("past_k", key), "input key cache");
            check(ex.input("past_v", value), "input value cache");
            key.release();
            value.release();
        }
        ncnn::Mat out;
        // type=1 is required for opaque cache storage and reserved capacity.
        check(ex.extract("out_k", key, 1), "extract key cache");
        check(ex.extract("out_v", value, 1), "extract value cache");
        check(ex.extract("out", out), "extract output");
        key_buffers += old_key != key.data;
        value_buffers += old_value != value.data;
        if (dedicated && (key.allocator != &allocator || value.allocator != &allocator))
            throw std::runtime_error("CPU cache escaped the session allocator");
        return out;
    }

    void reset() override
    {
        key.release();
        value.release();
        key_buffers = value_buffers = 0;
    }
};

#if NCNN_VULKAN
struct GpuSession final : Session {
    ncnn::VulkanDevice* device;
    ncnn::VkBlobAllocator blob_allocator, cache_allocator;
    ncnn::VkStagingAllocator staging_allocator;
    ncnn::Net net;
    ncnn::VkMat key, value;
    bool dedicated;

    GpuSession(ncnn::VulkanDevice* dev, const std::string& precision, bool use_allocator)
        : device(dev), blob_allocator(dev), cache_allocator(dev), staging_allocator(dev), dedicated(use_allocator)
    {
        configure(net.opt, precision);
        net.opt.use_vulkan_compute = true;
        net.set_vulkan_device(device);
        check(net.load_param_mem(graph), "load Vulkan graph");
        check(net.load_model(reinterpret_cast<const unsigned char*>(empty_model)), "load Vulkan model");
    }

    ncnn::Mat step(const Inputs& in, int hint) override
    {
        const void* old_key = key.data;
        const void* old_value = value.data;
        ncnn::VkCompute cmd(device);
        ncnn::Extractor ex = net.create_extractor();
        ex.set_blob_vkallocator(&blob_allocator);
        ex.set_workspace_vkallocator(&blob_allocator);
        ex.set_staging_vkallocator(&staging_allocator);
        if (dedicated)
            ex.set_kvcache_vkallocator(&cache_allocator);
        ex.set_kvcache_max_seqlen_hint(hint);
        feed(ex, in);
        if (!key.empty())
        {
            check(ex.input("past_k", key), "input Vulkan key cache");
            check(ex.input("past_v", value), "input Vulkan value cache");
            key.release();
            value.release();
        }
        ncnn::VkMat out_gpu;
        check(ex.extract("out_k", key, cmd), "extract Vulkan key cache");
        check(ex.extract("out_v", value, cmd), "extract Vulkan value cache");
        check(ex.extract("out", out_gpu, cmd), "extract Vulkan output");
        ncnn::Option transfer = net.opt;
        transfer.blob_vkallocator = &blob_allocator;
        transfer.workspace_vkallocator = &blob_allocator;
        transfer.staging_vkallocator = &staging_allocator;
        ncnn::Mat out;
        cmd.record_download(out_gpu, out, transfer);
        check(cmd.submit_and_wait(), "submit Vulkan step");
        key_buffers += old_key != key.data;
        value_buffers += old_value != value.data;
        if (dedicated && (key.allocator != &cache_allocator || value.allocator != &cache_allocator))
            throw std::runtime_error("Vulkan cache escaped the session allocator");
        return out;
    }

    void reset() override
    {
        key.release();
        value.release();
        key_buffers = value_buffers = 0;
    }
};
#endif

struct Error {
    double maximum = 0;
    double squared = 0;
    double reference_squared = 0;
    void accumulate(const ncnn::Mat& expected, const ncnn::Mat& actual)
    {
        if (actual.w != expected.w || actual.h != expected.h || actual.c != expected.c
            || actual.elempack != 1 || actual.elemsize != 4)
            throw std::runtime_error("output shape, packing or dtype mismatch");
        for (int h = 0; h < expected.c; ++h)
        {
            const ncnn::Mat a = expected.channel(h), b = actual.channel(h);
            for (int p = 0; p < expected.h; ++p)
                for (int d = 0; d < expected.w; ++d)
                {
                    const double ref = a.row(p)[d], got = b.row(p)[d];
                    if (!std::isfinite(got))
                        throw std::runtime_error("non-finite output");
                    const double delta = got - ref;
                    maximum = std::max(maximum, std::abs(delta));
                    squared += delta * delta;
                    reference_squared += ref * ref;
                }
        }
    }
    double nrmse() const { return std::sqrt(squared / std::max(reference_squared, 1e-30)); }
};

void run_case(Session& session, const std::string& label, const std::string& name,
              const std::vector<int>& lengths, int hint, unsigned seed,
              double atol, double relative_limit, bool expect_reuse)
{
    session.reset();
    int start = 0;
    Error error;
    for (int len : lengths)
    {
        const Inputs in = inputs(len, start, seed);
        const ncnn::Mat expected = reference(in, start, seed);
        error.accumulate(expected, session.step(in, hint));
        start += len;
    }
    const bool passed = error.maximum <= atol && error.nrmse() <= relative_limit
                     && (!expect_reuse || (session.key_buffers == 1 && session.value_buffers == 1));
    std::cout << std::setprecision(9)
              << "{\"configuration\":\"" << label << "\",\"case\":\"" << name
              << "\",\"tokens\":" << start << ",\"steps\":" << lengths.size()
              << ",\"hint\":" << hint << ",\"key_backing_allocations\":" << session.key_buffers
              << ",\"value_backing_allocations\":" << session.value_buffers
              << ",\"max_abs_error\":" << error.maximum << ",\"nrmse\":" << error.nrmse()
              << ",\"atol\":" << atol << ",\"nrmse_limit\":" << relative_limit
              << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
    if (!passed)
        throw std::runtime_error(label + ": " + name + " failed");
}

void suite(Session& session, const std::string& label, const std::string& precision, bool dedicated)
{
    // These are operator smoke-test gates, not full-model quality thresholds.
    const double atol = precision == "bf16" ? 0.015 : precision == "fp16" ? 0.002 : 0.00002;
    const double rel = precision == "bf16" ? 0.03 : precision == "fp16" ? 0.005 : 0.00002;
    run_case(session, label, "prefill_decode_reuse", {32, 1, 1, 1, 1, 1, 1, 1, 1}, 128, 11, atol, rel, dedicated);
    run_case(session, label, "reset_and_growth", {5, 1, 1, 1, 9, 17, 31}, 8, 83, atol, rel, false);
    run_case(session, label, "fresh_prompt_after_reset", {7, 1, 1}, 16, 2026, atol, rel, dedicated);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || std::string(argv[1]) != "--backend"
        || (std::string(argv[2]) != "cpu" && std::string(argv[2]) != "vulkan"))
    {
        std::cerr << "Usage: ernie-attention-probe --backend cpu|vulkan\n";
        return 2;
    }
    const std::string backend = argv[2];
    std::cout << "{\"ncnn_revision\":\"" << ERNIE_NCNN_REVISION
              << "\",\"q_heads\":32,\"kv_heads\":8,\"head_dim\":128,\"backend\":\""
              << backend << "\"}\n";
    int result = 0;
    try
    {
        if (backend == "cpu")
        {
            CpuSession dedicated(true);
            suite(dedicated, "cpu_fp32_dedicated", "fp32", true);
            CpuSession automatic(false);
            suite(automatic, "cpu_fp32_default", "fp32", false);
        }
        else
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() == 0)
            {
                std::cerr << "SKIP: no Vulkan device available\n";
                ncnn::destroy_gpu_instance();
                return 77;
            }
            ncnn::VulkanDevice* device = ncnn::get_gpu_device();
            std::cerr << "Vulkan device: " << device->info.device_name() << '\n';
            for (const std::string precision : {"fp32", "fp16", "bf16"})
            {
                if (precision == "fp16" && !device->info.support_fp16_storage())
                {
                    std::cerr << "SKIP: fp16 storage unavailable\n";
                    continue;
                }
                if (precision == "bf16" && !device->info.support_bf16_storage())
                {
                    std::cerr << "SKIP: bf16 storage unavailable\n";
                    continue;
                }
                GpuSession dedicated(device, precision, true);
                suite(dedicated, "vulkan_" + precision + "_dedicated", precision, true);
                GpuSession baseline(device, precision, false);
                suite(baseline, "vulkan_" + precision + "_default", precision, false);
            }
#else
            std::cerr << "SKIP: compiled without Vulkan\n";
            return 77;
#endif
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        result = 1;
    }
#if NCNN_VULKAN
    if (backend == "vulkan")
        ncnn::destroy_gpu_instance();
#endif
    return result;
}

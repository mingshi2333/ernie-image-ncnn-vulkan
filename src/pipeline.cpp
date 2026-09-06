// SPDX-License-Identifier: MIT
#include "conditioning.h"
#include "denoiser.h"
#include "model_config.h"
#include "prompt_enhancer.h"
#include "tensor_io.h"
#include "text_encoder.h"
#include "tokenizer.h"
#include "vae.h"
#include <ernie/pipeline.h>
#if NCNN_VULKAN
#include "pipelinecache.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;
namespace ernie
{
namespace
{
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(int value, const char *action)
{
    if (value)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(value));
}
class GpuContext
{
  public:
    explicit GpuContext(bool enabled) : enabled_(enabled)
    {
        if (!enabled_)
            return;
#if NCNN_VULKAN
        ncnn::create_gpu_instance();
#else
        throw std::runtime_error("Built without Vulkan");
#endif
    }
    ~GpuContext()
    {
#if NCNN_VULKAN
        if (enabled_)
            ncnn::destroy_gpu_instance();
#endif
    }

  private:
    bool enabled_;
};
void validate_request(const GenerationRequest &r)
{
    if (r.model.empty() || r.steps < 1 || r.steps > 1000 || (r.device != "cpu" && r.device != "vulkan") ||
        (r.precision != "fp32" && r.precision != "fp16" && r.precision != "bf16") ||
        (r.device == "cpu" && r.precision != "fp32") || (r.vae_device != "cpu" && r.vae_device != "vulkan") ||
        (r.vae_convolution != "sgemm" && r.vae_convolution != "direct"))
        throw std::invalid_argument("Invalid generation request");
    if (bool(r.width) != bool(r.height))
        throw std::invalid_argument("Specify width and height together");
    if (r.width &&
        (r.width < 16 || r.height < 16 || r.width > 2048 || r.height > 2048 || r.width % 16 || r.height % 16))
        throw std::invalid_argument("Width and height must be multiples of 16 in [16,2048]");
    if (!r.trace.empty() && fs::exists(r.trace))
        throw std::invalid_argument("Use a new trace directory");
    if (!r.pe_model.empty() && !r.embeddings.empty())
        throw std::invalid_argument("PE and precomputed embeddings cannot be combined");
}
void trace_text(const fs::path &path, const std::string &text)
{
    if (fs::exists(path))
        throw std::runtime_error("Trace file exists");
    std::ofstream file(path, std::ios::binary);
    if (!file.write(text.data(), text.size()))
        throw std::runtime_error("Cannot write prompt trace");
}
RgbImage rgb_image(const ncnn::Mat &decoded)
{
    if (decoded.empty() || decoded.dims != 3 || decoded.c != 3 || decoded.elempack != 1 ||
        decoded.elemsize != 4u)
        throw std::runtime_error("VAE output must be pack1 RGB FP32");
    RgbImage image{decoded.w, decoded.h, std::vector<uint8_t>(size_t(decoded.w) * decoded.h * 3)};
    for (int c = 0; c < 3; ++c)
    {
        const float *channel = decoded.channel(c);
        for (int i = 0; i < decoded.w * decoded.h; ++i)
        {
            if (!std::isfinite(channel[i]))
                throw std::runtime_error("VAE produced non-finite pixels");
            const float normalized = std::clamp(channel[i] / 2.f + .5f, 0.f, 1.f);
            image.pixels[size_t(i) * 3 + c] = static_cast<unsigned char>(std::nearbyint(normalized * 255.f));
        }
    }
    return image;
}
ncnn::Mat run_dit(const fs::path &root, const ncnn::Mat &initial, const std::vector<ncnn::Mat> &constants,
                  const ncnn::Option &cpu, const GenerationRequest &request, const ProgressCallback &notify)
{
    const auto &backend = request.device, &precision = request.precision;
    const int steps = request.steps;
    const fs::path trace(request.trace);
    ernie::DenoiseModel dit;
    dit.input_head = (root / "dit/input").string();
    dit.output_head = (root / "dit/output").string();
    for (int i = 0; i < 36; ++i)
        dit.blocks.push_back((root / "dit" / numbered("block-", i)).string());
    std::vector<ernie::DenoiseStepStats> stats;
    auto progress = [&](size_t i)
    {
        if (notify)
            notify({"denoise", int(i + 1), steps, stats.at(i).elapsed_seconds});
    };
    ncnn::Mat latent;
    if (backend == "cpu")
        latent = ernie::denoise(dit, initial, constants, steps, cpu, stats,
                                [&](size_t i, const ncnn::Mat &prediction, const ncnn::Mat &sample)
                                {
                                    if (!trace.empty())
                                    {
                                        write_tensor(trace / ("prediction-" + std::to_string(i) + ".f32"),
                                                     prediction);
                                        write_tensor(trace / ("step-" + std::to_string(i) + ".f32"), sample);
                                    }
                                    progress(i);
                                });
    else
    {
#if NCNN_VULKAN
        if (ncnn::get_gpu_count() < 1)
            throw std::runtime_error("No Vulkan device");
        const auto *device = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
        if (precision == "fp16" && !device->info.support_fp16_storage())
            throw std::runtime_error("Device lacks FP16 storage");
        if (precision == "bf16" && !device->info.support_bf16_storage())
            throw std::runtime_error("Device lacks BF16 storage");
        ncnn::PipelineCache cache(device);
        ncnn::VkBlobAllocator blobs(device);
        ncnn::VkStagingAllocator staging(device);
        ncnn::Option option = cpu;
        option.use_vulkan_compute = true;
        option.use_fp16_storage = precision == "fp16";
        option.use_bf16_storage = precision == "bf16";
        option.blob_vkallocator = option.workspace_vkallocator = &blobs;
        option.staging_vkallocator = &staging;
        option.pipeline_cache = &cache;
        ncnn::Option high = option;
        high.use_fp16_storage = high.use_bf16_storage = false;
        high.use_packing_layout = false;
        ncnn::VkMat gpu_initial;
        std::vector<ncnn::VkMat> gpu_constants(constants.size());
        {
            ncnn::VkCompute upload(device);
            ncnn::VkMat packed;
            upload.record_upload(initial, packed, high);
            device->convert_packing(packed, gpu_initial, 1, 1, upload, high);
            for (size_t i = 0; i < constants.size(); ++i)
                upload.record_upload(constants[i], gpu_constants[i], option);
            check(upload.submit_and_wait(), "Upload conditioning");
        }
        const auto gpu_latent =
            ernie::denoise(dit, gpu_initial, gpu_constants, steps, device, option, stats,
                           [&](size_t i, const ncnn::VkMat &prediction, const ncnn::VkMat &sample)
                           {
                               if (!trace.empty())
                               {
                                   ncnn::Mat p, x;
                                   ncnn::VkCompute download(device);
                                   download.record_download(prediction, p, high);
                                   download.record_download(sample, x, high);
                                   check(download.submit_and_wait(), "Download trace");
                                   write_tensor(trace / ("prediction-" + std::to_string(i) + ".f32"), p);
                                   write_tensor(trace / ("step-" + std::to_string(i) + ".f32"), x);
                               }
                               progress(i);
                           });
        ncnn::VkCompute download(device);
        download.record_download(gpu_latent, latent, high);
        check(download.submit_and_wait(), "Download final latent");
#else
        throw std::runtime_error("Built without Vulkan");
#endif
    }
    return latent;
}
} // namespace

void verify_model(const std::string &directory)
{
    model_config(fs::path(directory) / "model.cfg");
    verify_package(directory);
}
void verify_pe_model(const std::string &directory)
{
    if (!fs::is_regular_file(fs::path(directory) / "pe.cfg"))
        throw std::invalid_argument("Cannot open pe.cfg");
    verify_package(directory);
}

GenerationResult generate(const GenerationRequest &r, const ProgressCallback &notify)
{
    validate_request(r);
    const auto start = Clock::now();
    const fs::path root(r.model), trace(r.trace);
    const auto cfg = model_config(root / "model.cfg");
    const int w = cfg.packed_width, h = cfg.packed_height, bucket = cfg.text_bucket;
    if (r.width && (r.width != 16 * w || r.height != 16 * h))
        throw std::invalid_argument("Requested resolution differs from this static model bucket; "
                                    "prepare a matching package with tools/prepare_variant.py");
    verify_package(r.model);
    if (notify)
        notify({"verify", 1, 1, elapsed(start)});
    if (!trace.empty())
        fs::create_directories(trace);
    GenerationResult result;
    result.prompt = r.prompt;
    if (!r.pe_model.empty())
    {
        const auto enhanced = enhance_prompt(r.pe_model, r.prompt, w * 16, h * 16, r.pe,
                                             [&](const char *phase, int current, int total)
                                             {
                                                 if (notify)
                                                     notify({std::string("pe-") + phase, current, total, 0});
                                             });
        result.prompt = enhanced.text;
        result.pe_eos = enhanced.eos;
        result.pe_generated_tokens = enhanced.generated_ids.size();
        if (!trace.empty())
        {
            trace_text(trace / "input-prompt.txt", r.prompt);
            trace_text(trace / "enhanced-prompt.txt", enhanced.text);
            std::string ids;
            for (auto id : enhanced.generated_ids)
                ids += std::to_string(id) + '\n';
            trace_text(trace / "pe-ids.txt", ids);
        }
    }
    Tokenizer tokenizer((root / "tokenizer").string());
    const auto ids = tokenizer.encode(result.prompt);
    if (ids.empty() || ids.size() > size_t(bucket))
        throw std::invalid_argument("Prompt exceeds this model's " + std::to_string(bucket) +
                                    " token bucket");
    result.token_ids = ids;
    if (!trace.empty())
    {
        trace_text(trace / "prompt.txt", result.prompt);
        std::string tokens;
        for (auto id : ids)
            tokens += std::to_string(id) + '\n';
        trace_text(trace / "ids.txt", tokens);
    }
    ncnn::Option cpu;
    cpu.num_threads = 4;
    cpu.use_vulkan_compute = false;
    cpu.use_fp16_storage = cpu.use_fp16_packed = cpu.use_fp16_arithmetic = cpu.use_bf16_storage =
        cpu.use_bf16_packed = false;
    ncnn::Mat text;
    const auto text_start = Clock::now();
    if (!r.embeddings.empty())
        text = read_tensor(r.embeddings, 3072, int(ids.size())).reshape(3072, int(ids.size()));
    else
    {
        const auto embedded = text_embeddings((root / "text/embeddings.bf16").string(), ids, bucket);
        const auto constants = text_constants((root / "text/rope-inv-freq.f32").string(), bucket);
        std::vector<std::string> models;
        for (int i = 0; i < 25; ++i)
            models.push_back((root / "text" / numbered("block-", i)).string());
        BlockSequenceStats stats;
        const auto encoded = run_text_blocks(models, embedded, constants, cpu, stats);
        text = encoded.row_range(0, int(ids.size())).clone();
    }
    if (notify)
        notify({"text", int(ids.size()), bucket, elapsed(text_start)});
    if (!trace.empty())
        write_tensor(trace / "text.f32", text);
    const auto padded = pad_text(text, cfg.dit_text_tokens);
    const auto rotary =
        dit_constants((root / "dit/rope-inv-freq.f32").string(), w, h, int(ids.size()), cfg.dit_text_tokens);
    const std::vector<ncnn::Mat> constants{padded, rotary[0], rotary[1], rotary[2]};
    ncnn::Mat initial;
    if (!r.latent.empty())
        initial = read_tensor(r.latent, w, h, 128);
    else
    {
        initial.create(w, h, 128);
        if (initial.empty())
            throw std::bad_alloc();
        std::mt19937 engine(r.seed);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (int c = 0; c < 128; ++c)
        {
            float *values = initial.channel(c);
            for (int j = 0; j < w * h; ++j)
                values[j] = normal(engine);
        }
    }
    if (!trace.empty())
    {
        write_tensor(trace / "initial.f32", initial);
        write_tensor(trace / "padded-text.f32", padded);
        for (size_t i = 0; i < rotary.size(); ++i)
            write_tensor(trace / ("constant-" + std::to_string(i) + ".f32"), rotary[i]);
    }
    GpuContext gpu(r.device == "vulkan" || r.vae_device == "vulkan");
    const auto latent = run_dit(root, initial, constants, cpu, r, notify);
    const auto mean = read_tensor(root / "vae/bn-mean.f32", 128),
               variance = read_tensor(root / "vae/bn-variance.f32", 128);
    const auto unpacked = unpack_for_vae(latent, mean, variance, 4);
    if (!trace.empty())
    {
        write_tensor(trace / "final.f32", latent);
        write_tensor(trace / "unpacked.f32", unpacked);
    }
    const auto vae_start = Clock::now();
    const auto decoded = decode_vae((root / "vae").string(), unpacked, cpu, r.vae_device, r.vae_convolution);
    if (decoded.w != w * 16 || decoded.h != h * 16)
        throw std::runtime_error("Decoded resolution differs from model");
    if (!trace.empty())
        write_tensor(trace / "decoded.f32", decoded);
    result.image = rgb_image(decoded);
    result.vae_seconds = elapsed(vae_start);
    result.elapsed_seconds = elapsed(start);
    return result;
}
} // namespace ernie

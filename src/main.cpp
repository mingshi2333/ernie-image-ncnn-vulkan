// SPDX-License-Identifier: MIT
#include "conditioning.h"
#include "denoiser.h"
#include "ernie_gelu.h"
#include "text_encoder.h"
#include "tokenizer.h"
#if NCNN_VULKAN
#include "pipelinecache.h"
#endif
#include <png.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace
{
void check(int value, const char *action)
{
    if (value)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(value));
}
ncnn::Mat read(const fs::path &path, int w, int h = 1, int c = 1)
{
    if (fs::file_size(path) != size_t(w) * h * c * 4)
        throw std::runtime_error("Wrong tensor size: " + path.string());
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
            throw std::runtime_error("Tensor contains non-finite values: " + path.string());
    }
    return out;
}
void write(const fs::path &path, const ncnn::Mat &value)
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
std::map<std::string, int> model_config(const fs::path &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open model.cfg");
    std::map<std::string, int> values;
    std::string key, raw;
    while (file >> key)
    {
        if (!(file >> raw) || values.count(key))
            throw std::runtime_error("Malformed or duplicate model.cfg entry");
        size_t used = 0;
        const int value = std::stoi(raw, &used);
        if (used != raw.size())
            throw std::runtime_error("Invalid model.cfg integer");
        values[key] = value;
    }
    for (const auto *name :
         {"packed_width", "packed_height", "text_bucket", "dit_text_tokens", "text_layers", "dit_layers"})
        if (!values.count(name))
            throw std::runtime_error("Missing model.cfg field");
    const int w = values.at("packed_width"), h = values.at("packed_height"), t = values.at("dit_text_tokens"),
              bucket = values.at("text_bucket");
    if (values.size() != 6 || w < 1 || h < 1 || w > 128 || h > 128 || bucket < 1 || bucket > 2048 ||
        t < bucket || t > 2048 || w * h + t > 6144 || values.at("text_layers") != 25 ||
        values.at("dit_layers") != 36)
        throw std::runtime_error("Unsupported model configuration");
    return values;
}
void png_write(const fs::path &path, const ncnn::Mat &decoded)
{
    if (decoded.empty() || decoded.dims != 3 || decoded.c != 3 || decoded.elempack != 1 ||
        decoded.elemsize != 4u)
        throw std::runtime_error("VAE output must be pack1 RGB FP32");
    std::vector<unsigned char> pixels(size_t(decoded.w) * decoded.h * 3);
    for (int c = 0; c < 3; ++c)
    {
        const float *channel = decoded.channel(c);
        for (int i = 0; i < decoded.w * decoded.h; ++i)
        {
            if (!std::isfinite(channel[i]))
                throw std::runtime_error("VAE produced non-finite pixels");
            const float normalized = std::clamp(channel[i] / 2.f + .5f, 0.f, 1.f);
            pixels[size_t(i) * 3 + c] = static_cast<unsigned char>(std::nearbyint(normalized * 255.f));
        }
    }
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = decoded.w;
    image.height = decoded.h;
    image.format = PNG_FORMAT_RGB;
    if (!png_image_write_to_file(&image, path.string().c_str(), 0, pixels.data(), 0, nullptr))
        throw std::runtime_error(std::string("PNG write failed: ") + image.message);
}
} // namespace

int main(int argc, char **argv)
{
    fs::path root, output, latent_path, embeddings_path, trace;
    std::string prompt, backend = "vulkan", precision = "fp16", vae_backend = "cpu", vae_convolution = "direct";
    bool have_prompt = false, verify_only = false;
    int steps = 8, status = 0;
    uint32_t seed = 42;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--verify-model")
            {
                verify_only = true;
                continue;
            }
            if (flag == "--help")
            {
                std::cout << "ernie-image --model DIR --prompt TEXT --output NEW.png [--device cpu|vulkan] "
                             "[--precision fp32|fp16] [--vae-device cpu|vulkan] [--vae-convolution direct|sgemm]\n"
                          << "            [--seed N] [--steps N] [--latent FILE.f32] [--embeddings FILE.f32] "
                             "[--trace-dir NEWDIR]\n"
                          << "ernie-image --model DIR --verify-model\n"
                          << "Experimental fixed model bucket; FP32 text encoder, Euler master latent and "
                             "FP32 VAE (CPU default). PE off, CFG=1.\n";
                return 0;
            }
            if (++i == argc)
                throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[i];
            if (flag == "--model")
                root = value;
            else if (flag == "--output")
                output = value;
            else if (flag == "--vae-convolution")
                vae_convolution = value;
            else if (flag == "--prompt")
            {
                prompt = value;
                have_prompt = true;
            }
            else if (flag == "--device")
                backend = value;
            else if (flag == "--precision")
                precision = value;
            else if (flag == "--vae-device")
                vae_backend = value;
            else if (flag == "--latent")
                latent_path = value;
            else if (flag == "--embeddings")
                embeddings_path = value;
            else if (flag == "--trace-dir")
                trace = value;
            else if (flag == "--seed" || flag == "--steps")
            {
                size_t used = 0;
                const auto number = std::stoull(value, &used);
                if (used != value.size() || value.empty() || value[0] == '-' || number > UINT32_MAX)
                    throw std::invalid_argument("Invalid integer");
                if (flag == "--seed")
                    seed = uint32_t(number);
                else
                {
                    if (number < 1 || number > 1000)
                        throw std::invalid_argument("Steps must be in [1,1000]");
                    steps = int(number);
                }
            }
            else
                throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (root.empty() || (!verify_only && (output.empty() || !have_prompt || fs::exists(output))) ||
            (!trace.empty() && fs::exists(trace)) || (backend != "cpu" && backend != "vulkan") ||
            (precision != "fp32" && precision != "fp16") || (backend == "cpu" && precision != "fp32") ||
            (vae_backend != "cpu" && vae_backend != "vulkan") ||
            (vae_convolution != "sgemm" && vae_convolution != "direct"))
            throw std::invalid_argument("Invalid request; see --help and use new output paths");
        const auto total_start = std::chrono::steady_clock::now();
        const auto cfg = model_config(root / "model.cfg");
        ernie::verify_package(root.string());
        std::cout << "Model verified: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count()
                  << " s" << std::endl;
        if (verify_only)
            return 0;
        const int w = cfg.at("packed_width"), h = cfg.at("packed_height"), bucket = cfg.at("text_bucket");
        if (!trace.empty())
            fs::create_directories(trace);
        ernie::Tokenizer tokenizer((root / "tokenizer").string());
        const auto ids = tokenizer.encode(prompt);
        if (ids.empty() || ids.size() > size_t(bucket))
            throw std::invalid_argument("Prompt exceeds this model's " + std::to_string(bucket) +
                                        " token bucket");
        if (!trace.empty())
        {
            std::ofstream file(trace / "ids.txt");
            for (auto id : ids)
                file << id << '\n';
        }
        ncnn::Option cpu;
        cpu.num_threads = 4;
        cpu.use_vulkan_compute = false;
        cpu.use_fp16_storage = cpu.use_fp16_packed = cpu.use_fp16_arithmetic = cpu.use_bf16_storage =
            cpu.use_bf16_packed = false;
        ncnn::Mat text;
        const auto text_start = std::chrono::steady_clock::now();
        if (!embeddings_path.empty())
            text = read(embeddings_path, 3072, int(ids.size())).reshape(3072, int(ids.size()));
        else
        {
            const auto embedded =
                ernie::text_embeddings((root / "text/embeddings.bf16").string(), ids, bucket);
            const auto constants = ernie::text_constants((root / "text/rope-inv-freq.f32").string(), bucket);
            std::vector<std::string> models;
            for (int i = 0; i < 25; ++i)
                models.push_back((root / "text" / numbered("block-", i)).string());
            ernie::BlockSequenceStats stats;
            const auto encoded = ernie::run_text_blocks(models, embedded, constants, cpu, stats);
            text = encoded.row_range(0, int(ids.size())).clone();
        }
        std::cout << "Text conditioned: " << ids.size() << " tokens, "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - text_start).count()
                  << " s, embeddings " << (embeddings_path.empty() ? "computed" : "loaded") << std::endl;
        if (!trace.empty())
            write(trace / "text.f32", text);
        const auto padded = ernie::pad_text(text, cfg.at("dit_text_tokens"));
        const auto rotary = ernie::dit_constants((root / "dit/rope-inv-freq.f32").string(), w, h,
                                                 int(ids.size()), cfg.at("dit_text_tokens"));
        const std::vector<ncnn::Mat> constants{padded, rotary[0], rotary[1], rotary[2]};
        ncnn::Mat initial;
        if (!latent_path.empty())
            initial = read(latent_path, w, h, 128);
        else
        {
            initial.create(w, h, 128);
            if (initial.empty())
                throw std::bad_alloc();
            std::mt19937 engine(seed);
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
            write(trace / "initial.f32", initial);
            write(trace / "padded-text.f32", padded);
            for (size_t i = 0; i < rotary.size(); ++i)
                write(trace / ("constant-" + std::to_string(i) + ".f32"), rotary[i]);
        }
        ernie::DenoiseModel dit;
        dit.input_head = (root / "dit/input").string();
        dit.output_head = (root / "dit/output").string();
        for (int i = 0; i < 36; ++i)
            dit.blocks.push_back((root / "dit" / numbered("block-", i)).string());
        std::vector<ernie::DenoiseStepStats> stats;
        auto progress = [&](size_t i)
        {
            std::cout << "Denoise " << i + 1 << '/' << steps << ": " << stats.at(i).elapsed_seconds << " s"
                      << std::endl;
        };
        ncnn::Mat latent;
        if (backend == "cpu")
            latent = ernie::denoise(dit, initial, constants, steps, cpu, stats,
                                    [&](size_t i, const ncnn::Mat &prediction, const ncnn::Mat &sample)
                                    {
                                        if (!trace.empty())
                                        {
                                            write(trace / ("prediction-" + std::to_string(i) + ".f32"),
                                                  prediction);
                                            write(trace / ("step-" + std::to_string(i) + ".f32"), sample);
                                        }
                                        progress(i);
                                    });
        else
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1)
                throw std::runtime_error("No Vulkan device");
            const auto *device = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
            if (precision == "fp16" && !device->info.support_fp16_storage())
                throw std::runtime_error("Device lacks FP16 storage");
            ncnn::PipelineCache cache(device);
            ncnn::VkBlobAllocator blobs(device);
            ncnn::VkStagingAllocator staging(device);
            ncnn::Option option = cpu;
            option.use_vulkan_compute = true;
            option.use_fp16_storage = precision == "fp16";
            option.blob_vkallocator = option.workspace_vkallocator = &blobs;
            option.staging_vkallocator = &staging;
            option.pipeline_cache = &cache;
            ncnn::Option high = option;
            high.use_fp16_storage = false;
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
                                       write(trace / ("prediction-" + std::to_string(i) + ".f32"), p);
                                       write(trace / ("step-" + std::to_string(i) + ".f32"), x);
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
        const auto mean = read(root / "vae/bn-mean.f32", 128),
                   variance = read(root / "vae/bn-variance.f32", 128);
        const auto unpacked = ernie::unpack_for_vae(latent, mean, variance, 4);
        if (!trace.empty())
        {
            write(trace / "final.f32", latent);
            write(trace / "unpacked.f32", unpacked);
        }
        ncnn::Mat decoded;
        const auto vae_start = std::chrono::steady_clock::now();
        {
            ncnn::Net vae;
            vae.opt = cpu;
            vae.opt.use_winograd_convolution = false;
            vae.opt.use_sgemm_convolution = vae_convolution == "sgemm";
#if NCNN_VULKAN
            if (vae_backend == "vulkan")
            {
                if (backend != "vulkan")
                    ncnn::create_gpu_instance();
                if (ncnn::get_gpu_count() < 1)
                    throw std::runtime_error("No Vulkan device for VAE");
                vae.opt.use_vulkan_compute = true;
                vae.set_vulkan_device(ncnn::get_default_gpu_index());
            }
#else
            if (vae_backend == "vulkan")
                throw std::runtime_error("Built without Vulkan VAE support");
#endif
            check(ernie::register_layers(vae), "Register layers");
            check(vae.load_param((root / "vae/head.ncnn.param").string().c_str()), "Load VAE graph");
            check(vae.load_model((root / "vae/head.ncnn.bin").string().c_str()), "Load VAE weights");
            if (vae_backend == "vulkan")
                for (const auto *layer : vae.layers())
                    if (!layer->support_vulkan && layer->type != "Input" && layer->type != "Split")
                        throw std::runtime_error("VAE contains a compute layer without Vulkan support");
            auto ex = vae.create_extractor();
            check(ex.input("in0", unpacked), "Input VAE latent");
            check(ex.extract("out0", decoded), "Decode VAE");
        }
        if (decoded.w != w * 16 || decoded.h != h * 16)
            throw std::runtime_error("Decoded resolution differs from model");
        if (!trace.empty())
            write(trace / "decoded.f32", decoded);
        if (output.has_parent_path())
            fs::create_directories(output.parent_path());
        png_write(output, decoded);
        std::cout << "VAE and PNG: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - vae_start).count()
                  << " s\n";
        std::cout << "Saved " << decoded.w << 'x' << decoded.h << " PNG: " << output.string() << '\n';
        std::cout << "Total: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count()
                  << " s\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
#if NCNN_VULKAN
    if (backend == "vulkan" || vae_backend == "vulkan")
        ncnn::destroy_gpu_instance();
#endif
    return status;
}

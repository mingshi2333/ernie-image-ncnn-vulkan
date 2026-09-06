// SPDX-License-Identifier: MIT
#include "text_encoder.h"
#include "ernie_gelu.h"
#include <cctype>
#include <memory>
#include <cstring>
#include <algorithm>
#if NCNN_VULKAN
#include "pipelinecache.h"
#endif
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
namespace fs = std::filesystem;
// Opt-in experiment: use upstream vector reduction for each row of MLP down.
// The portable graph must explicitly name this custom layer; default math is unchanged.
class DiagnosticVectorDown final : public ncnn::Layer
{
    std::unique_ptr<ncnn::Layer> inner{ncnn::create_layer("InnerProduct")};
public:
    DiagnosticVectorDown() { one_blob_only = true; support_packing = false; }
    int load_param(const ncnn::ParamDict& pd) override
    {
        if (!inner || pd.get(0, 0) != 3072 || pd.get(1, -1) != 0 || pd.get(2, 0) != 28311552)
            return -1;
        return inner->load_param(pd);
    }
    int load_model(const ncnn::ModelBin& mb) override { return inner->load_model(mb); }
    int create_pipeline(const ncnn::Option& opt) override { return inner->create_pipeline(opt); }
    int destroy_pipeline(const ncnn::Option& opt) override { return inner->destroy_pipeline(opt); }
    int forward(const ncnn::Mat& bottom, ncnn::Mat& top, const ncnn::Option& opt) const override
    {
        if (bottom.dims != 2 || bottom.w != 9216 || bottom.elempack != 1 || bottom.elemsize != 4u)
            return -1;
        top.create(3072, bottom.h, 4u, opt.blob_allocator);
        if (top.empty()) return -100;
        // Process every row, including padded rows. No prompt/index/ID condition.
        for (int r = 0; r < bottom.h; ++r)
        {
            ncnn::Mat row(9216, const_cast<float*>(bottom.row(r)));
            ncnn::Mat result, plain;
            if (inner->forward(row, result, opt)) return -1;
            ncnn::convert_packing(result, plain, 1, opt);
            if (plain.dims != 1 || plain.w != 3072 || plain.elemsize != 4u) return -1;
            std::memcpy(top.row(r), plain.data, 3072 * sizeof(float));
        }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(DiagnosticVectorDown)
static ncnn::Mat read(const fs::path &path, int w, int h)
{
    if (fs::file_size(path) != size_t(w) * h * 4)
        throw std::runtime_error("Wrong input length");
    ncnn::Mat value(w, h);
    if (value.empty())
        throw std::bad_alloc();
    std::ifstream file(path, std::ios::binary);
    if (!file.read(static_cast<char *>(value.data), size_t(w) * h * 4))
        throw std::runtime_error("Input read failed");
    return value;
}
// Diagnostic-only CPU extraction. No change to the production text path.
static void dump_trace(const fs::path &directory, const std::string &name,
                       const ncnn::Mat &tensor, int tokens, int valid)
{
    ncnn::Mat plain;
    ncnn::Option unpack;
    unpack.use_packing_layout = false;
    unpack.num_threads = 4;
    ncnn::convert_packing(tensor, plain, 1, unpack);
    if (plain.empty() || plain.elemsize != 4u || plain.elempack != 1 || plain.dims > 3)
        throw std::runtime_error("Unsupported trace layout");
    const int rows = plain.h == tokens ? valid : plain.h;
    const auto path = directory / (name + ".f32");
    if (fs::exists(path)) throw std::runtime_error("Trace exists");
    std::ofstream out(path, std::ios::binary);
    for (int c = 0; c < plain.c; ++c)
        out.write(static_cast<const char *>(plain.channel(c).data), size_t(plain.w) * rows * 4);
    if (!out) throw std::runtime_error("Trace write failed");
    std::ofstream meta(directory / (name + ".json"));
    meta << "{\"layout\":\"CHW\",\"shape\":[" << plain.c << "," << rows << "," << plain.w
         << "],\"storage_rows\":" << plain.h << ",\"valid_tokens\":" << valid << "}\n";
}
static ncnn::Mat trace_cpu(const std::vector<std::string> &models, const ncnn::Mat &input,
                          const std::vector<ncnn::Mat> &constants, const ncnn::Option &option,
                          const fs::path &directory, const std::vector<std::string> &blobs,
                          int tokens, int valid, bool vector_down)
{
    fs::create_directories(directory);
    dump_trace(directory, "embedding", input, tokens, valid);
    for (size_t j = 0; j < constants.size(); ++j)
        dump_trace(directory, "constant-" + std::to_string(j), constants[j], tokens, valid);
    ncnn::Mat current = input;
    for (size_t i = 0; i < models.size(); ++i)
    {
        ncnn::Net net;
        net.opt = option;
        if (vector_down && net.register_custom_layer("DiagnosticVectorDown", DiagnosticVectorDown_layer_creator))
            throw std::runtime_error("Cannot register diagnostic vector down");
        if (ernie::register_layers(net) ||
            net.load_param((fs::path(models[i]) / "text.ncnn.param").string().c_str()) ||
            net.load_model((fs::path(models[i]) / "text.ncnn.bin").string().c_str()))
            throw std::runtime_error("Trace model load failed");
        auto extract = [&](const std::string &name) {
            auto ex = net.create_extractor();
            if (ex.input("in0", current.clone())) throw std::runtime_error("Trace activation input failed");
            for (size_t j = 0; j < constants.size(); ++j)
                if (ex.input(("in" + std::to_string(j+1)).c_str(), constants[j]))
                    throw std::runtime_error("Trace constant input failed");
            ncnn::Mat value;
            if (ex.extract(name.c_str(), value)) throw std::runtime_error("Trace extraction failed: " + name);
            return value;
        };
        // Each requested internal blob uses a fresh extractor, so inspection
        // cannot change the free-running out0 extraction or allocator reuse.
        for (const auto &blob : blobs) dump_trace(directory, "blob-" + blob, extract(blob), tokens, valid);
        current = extract("out0");
        dump_trace(directory, "layer-" + std::to_string(i), current, tokens, valid);
    }
    return current;
}
int main(int argc, char **argv)
{
    std::vector<std::string> models;
    fs::path fixture, output, ids_path, embeddings, frequencies, trace;
    std::vector<std::string> trace_blobs;
    int tokens = 0, valid = 0, requested_valid = 0, status = 0;
    std::string backend = "cpu", precision = "fp32";
    bool vector_down = false;
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const std::string flag = argv[i];
            if (flag == "--diagnostic-vector-down") { vector_down = true; continue; }
            if (++i == argc)
                throw std::invalid_argument("Missing argument value");
            const std::string value = argv[i];
            if (flag == "--trace-dir")
                trace = value;
            else if (flag == "--trace-blob")
            {
                if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isalnum(c) || c == '_'; }))
                    throw std::invalid_argument("Unsafe trace blob name");
                trace_blobs.push_back(value);
            }
            else if (flag == "--model")
                models.push_back(value);
            else if (flag == "--fixture")
                fixture = value;
            else if (flag == "--output")
                output = value;
            else if (flag == "--ids")
                ids_path = value;
            else if (flag == "--embeddings")
                embeddings = value;
            else if (flag == "--frequencies")
                frequencies = value;
            else if (flag == "--backend")
                backend = value;
            else if (flag == "--precision")
                precision = value;
            else if (flag == "--valid-tokens")
                requested_valid = std::stoi(value);
            else if (flag == "--tokens")
            {
                size_t n = 0;
                tokens = std::stoi(value, &n);
                if (n != value.size())
                    throw std::invalid_argument("Invalid token count");
            }
            else
                throw std::invalid_argument("Unknown argument: " + flag);
        }
        if (models.empty() || models.size() > 25 || output.empty() || fs::exists(output) || tokens < 1 ||
            tokens > 2048 || (backend != "cpu" && backend != "vulkan") ||
            (precision != "fp32" && precision != "fp16" && precision != "bf16") ||
            (backend == "cpu" && precision != "fp32") || (fixture.empty() == ids_path.empty()))
            throw std::invalid_argument(
                "Require models, new output, token bucket and exactly one fixture/ids source");
        if (vector_down && (trace.empty() || backend != "cpu"))
            throw std::invalid_argument("Vector down requires CPU diagnostic trace");
        if ((!trace.empty() && (backend != "cpu" || fs::exists(trace))) ||
            (!trace_blobs.empty() && (trace.empty() || models.size() != 1)))
            throw std::invalid_argument("Trace requires new CPU directory; internal blobs require one model");
        ncnn::Mat input;
        std::vector<ncnn::Mat> constants;
        if (!fixture.empty())
        {
            input = read(fixture / "in0.f32", 3072, tokens);
            constants = {read(fixture / "in1.f32", 128, tokens), read(fixture / "in2.f32", 128, tokens),
                         read(fixture / "in3.f32", tokens, tokens)};
            if (requested_valid < 0 || requested_valid > tokens)
                throw std::invalid_argument("Invalid valid-token prefix");
            valid = requested_valid ? requested_valid : tokens;
        }
        else
        {
            std::ifstream file(ids_path);
            if (!file)
                throw std::runtime_error("Cannot open token IDs");
            std::vector<uint32_t> ids;
            std::string item;
            while (file >> item)
            {
                size_t consumed = 0;
                const auto value = std::stoull(item, &consumed);
                if (consumed != item.size() || value >= 131072 || ids.size() >= size_t(tokens))
                    throw std::invalid_argument("Invalid IDs or prompt exceeds bucket");
                ids.push_back(uint32_t(value));
            }
            if (!file.eof())
                throw std::runtime_error("Cannot read token IDs");
            input = ernie::text_embeddings(embeddings.string(), ids, tokens);
            constants = ernie::text_constants(frequencies.string(), tokens);
            valid = int(ids.size());
        }
        ncnn::Option option;
        option.num_threads = vector_down ? 2 : 4;
        option.use_vulkan_compute = backend == "vulkan";
        option.use_fp16_storage = precision == "fp16";
        option.use_bf16_storage = precision == "bf16";
        option.use_fp16_packed = option.use_fp16_arithmetic = option.use_bf16_packed = false;
        ernie::BlockSequenceStats stats;
        ncnn::Mat result;
        if (!trace.empty())
            result = trace_cpu(models, input, constants, option, trace, trace_blobs, tokens, valid, vector_down);
        else if (backend == "cpu")
            result = ernie::run_text_blocks(models, input, constants, option, stats);
        else
        {
#if NCNN_VULKAN
            ncnn::create_gpu_instance();
            if (ncnn::get_gpu_count() < 1)
                throw std::runtime_error("No Vulkan device");
            const int index = ncnn::get_default_gpu_index();
            const auto &info = ncnn::get_gpu_info(index);
            if ((precision == "fp16" && !info.support_fp16_storage()) ||
                (precision == "bf16" && !info.support_bf16_storage()))
                throw std::runtime_error("Unsupported storage precision");
            const auto *device = ncnn::get_gpu_device(index);
            ncnn::PipelineCache cache(device);
            ncnn::VkBlobAllocator blobs(device);
            ncnn::VkStagingAllocator staging(device);
            option.pipeline_cache = &cache;
            option.blob_vkallocator = option.workspace_vkallocator = &blobs;
            option.staging_vkallocator = &staging;
            ncnn::VkMat gpu_input;
            std::vector<ncnn::VkMat> gpu_constants(constants.size());
            {
                ncnn::VkCompute upload(device);
                upload.record_upload(input, gpu_input, option);
                for (size_t i = 0; i < constants.size(); ++i)
                    upload.record_upload(constants[i], gpu_constants[i], option);
                if (upload.submit_and_wait())
                    throw std::runtime_error("Text upload failed");
            }
            const auto gpu_output =
                ernie::run_text_blocks(models, gpu_input, gpu_constants, device, option, stats);
            ncnn::VkCompute download(device);
            ncnn::Option plain = option;
            plain.use_packing_layout = false;
            download.record_download(gpu_output, result, plain);
            if (download.submit_and_wait())
                throw std::runtime_error("Text download failed");
#else
            throw std::runtime_error("Built without Vulkan");
#endif
        }
        if (result.empty() || result.dims != 2 || result.w != 3072 || result.h != tokens ||
            result.elemsize != 4u || result.elempack != 1)
            throw std::runtime_error("Invalid text output layout");
        if (output.has_parent_path())
            fs::create_directories(output.parent_path());
        std::ofstream file(output, std::ios::binary);
        file.write(static_cast<char *>(result.data), size_t(valid) * 3072 * 4);
        if (!file)
            throw std::runtime_error("Text output write failed");
        std::cout << std::setprecision(12) << "{\"blocks\":" << models.size() << ",\"tokens\":" << tokens
                  << ",\"valid_tokens\":" << valid << ",\"backend\":\"" << backend << "\",\"precision\":\""
                  << precision << "\",\"load_seconds\":"
                  << std::accumulate(stats.load_seconds.begin(), stats.load_seconds.end(), 0.)
                  << ",\"compute_seconds\":"
                  << std::accumulate(stats.compute_seconds.begin(), stats.compute_seconds.end(), 0.)
                  << ",\"compute_submissions\":" << stats.compute_submissions << "}\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
#if NCNN_VULKAN
    if (backend == "vulkan")
        ncnn::destroy_gpu_instance();
#endif
    return status;
}

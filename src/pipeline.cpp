// SPDX-License-Identifier: MIT
#include "conditioning.h"
#include "denoiser.h"
#include "gpu_context.h"
#include "image_encoder.h"
#include "img2img.h"
#include "model_package.h"
#include "model_loading.h"
#include "prompt_enhancer.h"
#include "pipeline_metrics.h"
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
#include <iomanip>
#include <limits>
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
struct MetricsRecorder {
    ExecutionMetrics* metrics=nullptr;std::uint64_t event=1;
    void seconds(ExecutionPhase phase,double value) {
        if(!metrics || !(value>=0))return;
        const auto ns=std::uint64_t(value*1000000000.0);
        metrics->record_interval(event++,phase,0,ns);
    }
    void since(ExecutionPhase phase,Clock::time_point start) {if(metrics)seconds(phase,elapsed(start));}
    void submissions(std::uint64_t count) {if(metrics && count)metrics->record_submissions(event++,count);}
    void blocks(const BlockSequenceStats& stats,const char* component="text",int step=-1) {
        for(double x:stats.load_seconds)seconds(ExecutionPhase::ReadPrepare,x);
        for(double x:stats.compute_seconds)seconds(ExecutionPhase::Compute,x);
        submissions(std::uint64_t(stats.compute_submissions));
        if(metrics)for(const auto& d:stats.details)metrics->record_component(component,step,d.block,d.boundary,
            std::uint64_t(d.seconds*1000000000.0),d.status);
    }
    void denoise_step(const DenoiseStepStats& step,int absolute_step) {
        blocks(step.dit.blocks,"dit/block",absolute_step);
        if(metrics)for(const auto& d:step.dit.details)metrics->record_component(std::string("dit/")+d.component,
            absolute_step,-1,d.boundary,std::uint64_t(d.seconds*1000000000.0),d.status);
        if(!step.complete)return;
        seconds(ExecutionPhase::ReadPrepare,step.dit.input_head_seconds+step.dit.output_head_seconds);
        double accounted=step.dit.input_head_seconds+step.dit.output_head_seconds;
        for(double x:step.dit.blocks.load_seconds)accounted+=x;
        for(double x:step.dit.blocks.compute_seconds)accounted+=x;
        seconds(ExecutionPhase::Compute,std::max(0.,step.elapsed_seconds-accounted));
    }
    void vae(const VaeStats& stats) {
        if(metrics)for(const auto& d:stats.details)metrics->record_component("vae/decode",-1,-1,d.boundary,
            std::uint64_t(d.seconds*1000000000.0),d.status);
    }
};
void check(int value, const char *action)
{
    if (value)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(value));
}
void validate_request(const GenerationRequest &r)
{
    if (r.model.empty() || r.steps < 1 || r.steps > 1000 || (r.device != "cpu" && r.device != "vulkan") ||
        (r.precision != "fp32" && r.precision != "fp16" && r.precision != "bf16") ||
        (r.device == "cpu" && r.precision != "fp32") || (r.vae_device != "cpu" && r.vae_device != "vulkan") ||
        (r.vae_convolution != "sgemm" && r.vae_convolution != "direct"))
        throw std::invalid_argument("Invalid generation request");
    if (r.threads < 1 || r.threads > 256)
        throw std::invalid_argument("Threads must be in [1,256]");
    if (r.gpu_index < -1 || r.gpu_index > 63)
        throw std::invalid_argument("GPU index must be -1 or in [0,63]");
    if (r.gpu_index >= 0 && r.device != "vulkan" && r.vae_device != "vulkan")
        throw std::invalid_argument("GPU index requires a Vulkan generation or VAE device");
    if (r.text_device != "cpu")
        throw std::invalid_argument("Only CPU text encoding is currently supported");
    parse_weight_memory(r.dit_weights);
    request_mapped_model_loading(r.model_loading, false);
    if (r.device != "vulkan" && (r.dit_weights != "auto" || r.gpu_reserve_mib != 512))
        throw std::invalid_argument("DiT weight placement requires Vulkan generation");
    if ((r.dit_cache_mib || r.ram_reserve_mib != 3072) &&
        (r.device != "vulkan" || r.precision != "fp32" || r.dit_weights == "device"))
        throw std::invalid_argument("DiT RAM cache requires Vulkan FP32 and auto/host weights");
    if (!r.dit_cache_mib && r.ram_reserve_mib != 3072)
        throw std::invalid_argument("RAM reserve requires a nonzero DiT cache budget");
    if (!std::isfinite(r.strength) || r.strength < 0.f || r.strength > 1.f)
        throw std::invalid_argument("Strength must be finite and in [0,1]");
    if (bool(r.width) != bool(r.height))
        throw std::invalid_argument("Specify width and height together");
    if (r.width &&
        (r.width < 16 || r.height < 16 || r.width > 2048 || r.height > 2048 || r.width % 16 || r.height % 16))
        throw std::invalid_argument("Width and height must be multiples of 16 in [16,2048]");
    if (!r.trace.empty() && fs::exists(r.trace))
        throw std::invalid_argument("Use a new trace directory");
    if (!r.pe_model.empty() && !r.embeddings.empty())
        throw std::invalid_argument("PE and precomputed embeddings cannot be combined");
    if (r.text_down_vector && !r.embeddings.empty())
        throw std::invalid_argument("Vector text reduction requires native text encoding");
    if (r.input_image)
    {
        const auto &image = *r.input_image;
        if (image.width < 1 || image.height < 1 ||
            size_t(image.width) > std::numeric_limits<size_t>::max() / size_t(image.height) / 3 ||
            image.pixels.size() != size_t(image.width) * size_t(image.height) * 3)
            throw std::invalid_argument("Input image must contain exactly width*height*3 RGB bytes");
        if (r.width && (r.width != image.width || r.height != image.height))
            throw std::invalid_argument("Input image dimensions differ from the requested resolution");
        if (r.input_resize != "none" && r.input_resize != "stretch" && r.input_resize != "fit" && r.input_resize != "crop")
            throw std::invalid_argument("Invalid img2img resize identity");
        if (r.vae_device != "cpu")
            throw std::invalid_argument("The reviewed img2img VAE encoder is CPU-only");
        if (r.strength == 0.f && (!r.prompt.empty() || !r.pe_model.empty() || !r.embeddings.empty() || r.text_down_vector))
            throw std::invalid_argument("Strength-zero img2img does not consume prompt, PE, embeddings, or text reduction");
    }
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
ncnn::Mat run_dit(const ModelPackage &package, const ncnn::Mat &initial, const std::vector<ncnn::Mat> &constants,
                  const ncnn::Option &cpu, const GenerationRequest &request, const ProgressCallback &notify,
                  int start_step, MetricsRecorder& metrics, GenerationResult& result)
{
    const auto &backend = request.device, &precision = request.precision;
    const int steps = request.steps;
    const fs::path trace(request.trace);
    ernie::DenoiseModel dit;
    dit.input_head = package.component("dit/input/head.ncnn.param", "input");
    dit.output_head = package.component("dit/output/head.ncnn.param", "output");
    for (int i = 0; i < 36; ++i)
        dit.blocks.push_back(package.component("dit/" + numbered("block-", i) + "/block.ncnn.param", "dit"));
    std::vector<ernie::DenoiseStepStats> stats;
    size_t reported_steps=0;
    const int executed_steps = steps - start_step;
    auto progress = [&](size_t i)
    {
        if (notify)
            notify({"denoise", int(i) - start_step + 1, executed_steps,
                    stats.at(i - size_t(start_step)).elapsed_seconds});
    };
    ncnn::Mat latent;
    if (backend == "cpu")
        try { latent = ernie::denoise(dit, initial, constants, steps, cpu, stats,
                                [&](size_t i, const ncnn::Mat &prediction, const ncnn::Mat &sample)
                                {
                                    metrics.denoise_step(stats.at(i-size_t(start_step)),int(i));
                                    ++reported_steps;
                                    if (!trace.empty())
                                    {
                                        write_tensor(trace / ("prediction-" + std::to_string(i) + ".f32"),
                                                     prediction);
                                        write_tensor(trace / ("step-" + std::to_string(i) + ".f32"), sample);
                                    }
                                    progress(i);
                                }, start_step, metrics.metrics != nullptr); }
        catch (...) { for(size_t k=reported_steps;k<stats.size();++k)metrics.denoise_step(stats[k],start_step+int(k));throw; }
    else
    {
#if NCNN_VULKAN
        if (ncnn::get_gpu_count() < 1)
            throw std::runtime_error("No Vulkan device");
        const int gpu_index = request.gpu_index >= 0 ? request.gpu_index : ncnn::get_default_gpu_index();
        if (gpu_index < 0 || gpu_index >= ncnn::get_gpu_count())
            throw std::invalid_argument("Requested Vulkan GPU index is unavailable");
        const auto *device = ncnn::get_gpu_device(gpu_index);
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
            Clock::time_point transfer_start;if(metrics.metrics)transfer_start=Clock::now();
            ncnn::VkCompute upload(device);
            ncnn::VkMat packed;
            upload.record_upload(initial, packed, high);
            device->convert_packing(packed, gpu_initial, 1, 1, upload, high);
            for (size_t i = 0; i < constants.size(); ++i)
                upload.record_upload(constants[i], gpu_constants[i], option);
            check(upload.submit_and_wait(), "Upload conditioning");
            metrics.since(ExecutionPhase::Upload,transfer_start);metrics.submissions(1);
        }
        std::ofstream placement_trace;
        if (!trace.empty())
        {
            placement_trace.open(trace / "weight-placement.txt", std::ios::binary);
            if (!placement_trace) throw std::runtime_error("Cannot write weight placement trace");
        }
        WeightPlacement placement(parse_weight_memory(request.dit_weights),
            std::uint64_t(request.gpu_reserve_mib) * 1024 * 1024,
            vulkan_memory_budget_reader(device, gpu_initial.data->memory_type_index),
            [&](const ComponentFiles& files, const WeightPlacementDecision& decision) {
                if (trace.empty()) return;
                placement_trace << std::quoted(files.weight_path) << " requested=" << (decision.host ? "host" : "device")
                    << " reason=" << decision.reason << " available_bytes=";
                if (decision.available_bytes) placement_trace << *decision.available_bytes;
                else placement_trace << "unavailable";
                placement_trace << " estimated_weight_bytes=" << decision.weight_bytes
                    << " reserve_bytes=" << decision.reserve_bytes << '\n';
                if (!placement_trace) throw std::runtime_error("Cannot write weight placement trace");
            });
        std::unique_ptr<WeightSession> weight_session;
        if (request.dit_cache_mib)
            weight_session = std::make_unique<WeightSession>(WeightBudget{
                std::uint64_t(request.dit_cache_mib) * 1024 * 1024,
                std::uint64_t(request.ram_reserve_mib) * 1024 * 1024},
                host_memory_available_reader(), dit_host_weight_inspector(device));
        ncnn::VkMat gpu_latent;
        try { gpu_latent = ernie::denoise(dit, gpu_initial, gpu_constants, steps, device, option, stats,
                           [&](size_t i, const ncnn::VkMat &prediction, const ncnn::VkMat &sample)
                           {
                               metrics.denoise_step(stats.at(i-size_t(start_step)),int(i));
                               ++reported_steps;
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
                           }, start_step, metrics.metrics != nullptr, &placement, weight_session.get()); }
        catch (...) { for(size_t k=reported_steps;k<stats.size();++k)metrics.denoise_step(stats[k],start_step+int(k));throw; }
        result.host_weight_requests = placement.host_requests();
        result.device_weight_requests = placement.device_requests();
        result.unavailable_memory_queries = placement.unavailable_queries();
        if (weight_session)
        {
            const auto cache_stats = weight_session->stats();
            result.weight_cache_hits = cache_stats.hits;
            result.weight_cache_loads = cache_stats.loads;
            result.weight_cache_peak_bytes = cache_stats.peak_bytes;
            result.weight_cache_peak_nets = cache_stats.peak_nets;
            result.weight_cache_evictions = cache_stats.evictions;
            result.unavailable_host_memory_queries = cache_stats.unavailable_queries;
            if (!trace.empty())
                trace_text(trace / "weight-cache.txt", "hits=" + std::to_string(cache_stats.hits) +
                    "\nloads=" + std::to_string(cache_stats.loads) +
                    "\nadmissions=" + std::to_string(cache_stats.admissions) +
                    "\nevictions=" + std::to_string(cache_stats.evictions) +
                    "\npeak_charged_bytes=" + std::to_string(cache_stats.peak_bytes) +
                    "\npeak_nets=" + std::to_string(cache_stats.peak_nets) +
                    "\nunavailable_queries=" + std::to_string(cache_stats.unavailable_queries) + "\n");
            weight_session.reset(); // Release all DiT weights before VAE.
        }
        if (placement_trace.is_open())
        {
            placement_trace.close();
            if (!placement_trace) throw std::runtime_error("Cannot finish weight placement trace");
        }
        ncnn::VkCompute download(device);
        Clock::time_point transfer_start;if(metrics.metrics)transfer_start=Clock::now();
        download.record_download(gpu_latent, latent, high);
        check(download.submit_and_wait(), "Download final latent");
        metrics.since(ExecutionPhase::Download,transfer_start);metrics.submissions(1);
#else
        throw std::runtime_error("Built without Vulkan");
#endif
    }
    return latent;
}
} // namespace

void verify_model(const std::string &directory)
{
    const auto root = fs::path(directory);
    if (fs::is_regular_file(root / "model.cfg"))
        model_config(root / "model.cfg");
    else if (fs::exists(root / "pe.cfg"))
        throw std::invalid_argument("Use verify_pe_model for a prompt enhancer package");
    // The verifier checks every instance of a shared package without selecting
    // a generation resolution or reading any model weight into ncnn.
    verify_package(directory);
}
void verify_pe_model(const std::string &directory)
{
    if (!fs::is_regular_file(fs::path(directory) / "pe.cfg"))
        throw std::invalid_argument("Cannot open pe.cfg");
    verify_package(directory);
}

GenerationResult generate_impl(const GenerationRequest &r, const ProgressCallback &notify,ExecutionMetrics* collected)
{
    MetricsRecorder metrics{collected};
    validate_request(r);
    const auto start = Clock::now();
    const bool img2img = r.input_image.has_value();
    const bool denoise_image = img2img && r.strength > 0.f;
    const int request_width = r.width ? r.width : (img2img ? r.input_image->width : 0);
    const int request_height = r.height ? r.height : (img2img ? r.input_image->height : 0);
    // Initialize and validate the selected device before parsing model metadata
    // or loading PE/text/image weights.
    GpuContext gpu(((!img2img || denoise_image) && r.device == "vulkan") || r.vae_device == "vulkan", r.gpu_index);
    const fs::path trace(r.trace);
    ModelPackage package(r.model, request_width, request_height);
    const int w = package.config().packed_width, h = package.config().packed_height;
    if (notify)
        notify({"verify", 1, 1, elapsed(start)});
    metrics.since(ExecutionPhase::Verify,start);
    if (!trace.empty())
        fs::create_directories(trace);
    GenerationResult result;
    result.prompt = r.prompt;
    ncnn::Option cpu;
    cpu.num_threads = r.threads;
#if defined(ERNIE_EXPERIMENT_MAPPED_MODEL_LOADING)
    constexpr bool build_mapped_default = true;
#else
    constexpr bool build_mapped_default = false;
#endif
    // Each Net owns its mapping, including when retained in the RAM cache.
    // GPU stage options inherit the policy; ncnn may fall back to file reads.
    cpu.use_mapped_model_loading = request_mapped_model_loading(r.model_loading, build_mapped_default);
    result.mapped_model_loading_requested = cpu.use_mapped_model_loading;
    cpu.use_vulkan_compute = false;
    cpu.use_fp16_storage = cpu.use_fp16_packed = cpu.use_fp16_arithmetic = cpu.use_bf16_storage =
        cpu.use_bf16_packed = false;
    cpu.use_sgemm_convolution = false;
    cpu.use_winograd_convolution = false;
    if (!trace.empty())
        trace_text(trace / "model-loading.txt", std::string("requested=") +
                   (result.mapped_model_loading_requested ? "mapped" : "stdio") +
                   "\nscope=image text, DiT, VAE; excludes prompt enhancer\n");
    ncnn::Mat encoded;
    double encoder_seconds=0;
    if (img2img)
    {
        if (!package.has_file("vae/encoder.ncnn.param") || !package.has_file("vae/encoder.ncnn.bin"))
            throw std::invalid_argument("Selected model package has no reviewed img2img encoder");
        const auto encoder_start=Clock::now();
        const auto encoding=encode_vae(package.component("vae/encoder.ncnn.param","vae-encoder"),*r.input_image,cpu);
        encoder_seconds=elapsed(encoder_start);
        metrics.seconds(ExecutionPhase::ReadPrepare,encoder_seconds);
        encoded=encoding.normalized;
        if (!trace.empty())
        {
            std::ofstream input(trace / "input.rgb",std::ios::binary);
            if (!input.write(reinterpret_cast<const char *>(r.input_image->pixels.data()),
                             std::streamsize(r.input_image->pixels.size())))
                throw std::runtime_error("Cannot write img2img input trace");
            trace_text(trace / "img2img.txt", "source_width=" + std::to_string(r.input_source_width ? r.input_source_width : r.input_image->width) +
                       "\nsource_height=" + std::to_string(r.input_source_height ? r.input_source_height : r.input_image->height) +
                       "\nwidth=" + std::to_string(r.input_image->width) + "\nheight=" + std::to_string(r.input_image->height) + "\nstrength=" +
                       std::to_string(r.strength) + "\nresize=" + r.input_resize + "\nbackground=" +
                       std::to_string(r.input_resize_background[0]) + "," + std::to_string(r.input_resize_background[1]) +
                       "," + std::to_string(r.input_resize_background[2]) + "\nalpha_background=" +
                       std::to_string(r.input_alpha_background[0]) + "," + std::to_string(r.input_alpha_background[1]) +
                       "," + std::to_string(r.input_alpha_background[2]) + "\n");
            write_tensor(trace / "encoder-mean.f32",encoding.mean);
            write_tensor(trace / "encoder-packed.f32",encoding.packed);
            write_tensor(trace / "encoder-normalized.f32",encoded);
        }
        if (!denoise_image)
        {
            const auto mean=read_tensor(package.file("vae/bn-mean.f32"),128),
                       variance=read_tensor(package.file("vae/bn-variance.f32"),128);
            const auto unpacked=unpack_for_vae(encoded,mean,variance,r.threads);
            const auto vae_start=Clock::now();
            VaeStats vae_stats;
            ncnn::Mat decoded;
            try { decoded=decode_vae(package.component("vae/head.ncnn.param","vae"),unpacked,cpu,
                                    r.vae_device,r.vae_convolution,r.gpu_index,
                                    metrics.metrics ? &vae_stats : nullptr); }
            catch (...) { metrics.vae(vae_stats);throw; }
            metrics.vae(vae_stats);
            metrics.seconds(ExecutionPhase::ReadPrepare,elapsed(vae_start));
            if (decoded.w!=w*16 || decoded.h!=h*16)throw std::runtime_error("Decoded resolution differs from model");
            if (!trace.empty()){write_tensor(trace/"final.f32",encoded);write_tensor(trace/"unpacked.f32",unpacked);write_tensor(trace/"decoded.f32",decoded);}
            result.image=rgb_image(decoded);result.vae_seconds=encoder_seconds+elapsed(vae_start);result.elapsed_seconds=elapsed(start);
            return result;
        }
    }
    if (!r.pe_model.empty())
    {
        const auto enhanced = enhance_prompt(
            r.pe_model, r.prompt, w * 16, h * 16, r.pe,
            [&](const char *phase, int current, int total)
            {
                if (notify)
                    notify({std::string("pe-") + phase, current, total, 0});
            },
            {}, r.threads);
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
    Tokenizer tokenizer(package.file("tokenizer/tokenizer.json"),
                        package.file("tokenizer/tokenizer_config.json"));
    const auto ids = tokenizer.encode(result.prompt);
    package.select_text_tokens(ids.size());
    const auto cfg = package.config();
    const int bucket = cfg.text_bucket;
    std::optional<ShapePlan> shape;
    if (package.schema() == 3)
        shape = ShapePlan::create({{bucket}, cfg.dit_text_tokens}, w * 16, h * 16, ids.size());
    const bool host_weights = shape && shape->host_weights;
    if (r.text_down_vector && bucket != 32 && bucket != 64 && bucket != 2048)
        throw std::invalid_argument("Vector text reduction requires an independent 32, 64 or 2048 token graph");
    result.token_ids = ids;
    if (!trace.empty())
    {
        trace_text(trace / "prompt.txt", result.prompt);
        std::string tokens;
        for (auto id : ids)
            tokens += std::to_string(id) + '\n';
        trace_text(trace / "ids.txt", tokens);
        trace_text(trace / "shape.txt", "width=" + std::to_string(w * 16) +
                   "\nheight=" + std::to_string(h * 16) + "\nvalid_text_tokens=" + std::to_string(ids.size()) +
                   "\ntext_bucket=" + std::to_string(bucket) + "\ndit_text_tokens=" + std::to_string(cfg.dit_text_tokens) +
                   "\ntotal_tokens=" + std::to_string(w * h + cfg.dit_text_tokens) +
                   "\ndit_weights=" + (r.device == "cpu" ? std::string("host") : r.dit_weights) + "\n");
    }
    ncnn::Mat text;
    const auto text_start = Clock::now();
    if (!r.embeddings.empty())
        text = read_tensor(r.embeddings, 3072, int(ids.size())).reshape(3072, int(ids.size()));
    else
    {
        const auto embedded = text_embeddings(package.file("text/embeddings.bf16"), ids, bucket);
        const auto constants = text_constants(package.file("text/rope-inv-freq.f32"), bucket);
        std::vector<ComponentFiles> models;
        for (int i = 0; i < 25; ++i)
            models.push_back(package.component("text/" + numbered("block-", i) + "/text.ncnn.param", "text"));
        BlockSequenceStats stats;
        stats.collect_details = metrics.metrics != nullptr;
        ncnn::Mat encoded;
        try { encoded=run_text_blocks(models, embedded, constants, cpu, stats,
                                      r.text_down_vector ? TextDownMode::Vector : TextDownMode::Gemm); }
        catch (...) { metrics.blocks(stats);throw; }
        metrics.blocks(stats);
        text = encoded.row_range(0, int(ids.size())).clone();
    }
    if (notify)
        notify({"text", int(ids.size()), bucket, elapsed(text_start)});
    if (!trace.empty())
        write_tensor(trace / "text.f32", text);
    const auto padded = pad_text(text, cfg.dit_text_tokens);
    const auto rotary = shape ? dit_constants(package.file("dit/rope-inv-freq.f32"), *shape) :
        dit_constants(package.file("dit/rope-inv-freq.f32"), w, h, int(ids.size()), cfg.dit_text_tokens);
    const std::vector<ncnn::Mat> constants{padded, rotary[0], rotary[1], rotary[2]};
    ncnn::Mat noise;
    if (!r.latent.empty())
        noise = read_tensor(r.latent, w, h, 128);
    else
    {
        noise.create(w, h, 128);
        if (noise.empty())
            throw std::bad_alloc();
        std::mt19937 engine(r.seed);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (int c = 0; c < 128; ++c)
        {
            float *values = noise.channel(c);
            for (int j = 0; j < w * h; ++j)
                values[j] = normal(engine);
        }
    }
    int start_step=0;
    ncnn::Mat initial=noise;
    if (denoise_image)
    {
        auto image_start=make_img2img_start(encoded,noise,FlowSchedule::turbo(r.steps),r.strength,r.threads);
        initial=image_start.latent;start_step=image_start.start_step;
    }
    if (!trace.empty())
    {
        if (denoise_image)write_tensor(trace / "noise.f32",noise);
        write_tensor(trace / "initial.f32", initial);
        write_tensor(trace / "padded-text.f32", padded);
        for (size_t i = 0; i < rotary.size(); ++i)
            write_tensor(trace / ("constant-" + std::to_string(i) + ".f32"), rotary[i]);
    }
    auto dit_option = cpu;
    dit_option.use_weights_in_host_memory = host_weights;
    const auto latent = run_dit(package, initial, constants, dit_option, r, notify, start_step,metrics,result);
    const auto mean = read_tensor(package.file("vae/bn-mean.f32"), 128),
               variance = read_tensor(package.file("vae/bn-variance.f32"), 128);
    const auto unpacked = unpack_for_vae(latent, mean, variance, r.threads);
    if (!trace.empty())
    {
        write_tensor(trace / "final.f32", latent);
        write_tensor(trace / "unpacked.f32", unpacked);
    }
    const auto vae_start = Clock::now();
    VaeStats vae_stats;
    ncnn::Mat decoded;
    try { decoded=decode_vae(package.component("vae/head.ncnn.param", "vae"), unpacked, cpu,
                            r.vae_device, r.vae_convolution, r.gpu_index,
                            metrics.metrics ? &vae_stats : nullptr); }
    catch (...) { metrics.vae(vae_stats);throw; }
    metrics.vae(vae_stats);
    metrics.seconds(ExecutionPhase::ReadPrepare,elapsed(vae_start));
    if (decoded.w != w * 16 || decoded.h != h * 16)
        throw std::runtime_error("Decoded resolution differs from model");
    if (!trace.empty())
        write_tensor(trace / "decoded.f32", decoded);
    result.image = rgb_image(decoded);
    result.vae_seconds = encoder_seconds + elapsed(vae_start);
    result.elapsed_seconds = elapsed(start);
    return result;
}
GenerationResult generate(const GenerationRequest &r, const ProgressCallback &notify)
{
    return generate_impl(r,notify,nullptr);
}
GenerationResult generate_with_metrics(const GenerationRequest &r,const ProgressCallback& notify,ExecutionMetrics& metrics)
{
    return generate_impl(r,notify,&metrics);
}
} // namespace ernie

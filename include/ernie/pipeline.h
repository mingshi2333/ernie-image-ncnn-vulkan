// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ernie
{
struct PeOptions
{
    int max_tokens = 2048;
    float temperature = .6f, top_p = .95f;
    uint32_t seed = 42;
    bool greedy = false;
};

struct RgbImage
{
    int width = 0, height = 0;
    std::vector<uint8_t> pixels; // Interleaved RGB, 8 bits per channel.
};

struct GenerationRequest
{
    // All text and filesystem paths in this API use UTF-8 on every platform.
    std::string model, prompt;
    std::string device = "vulkan", precision = "fp16";
    std::string vae_device = "cpu", vae_convolution = "direct";
    int width = 0, height = 0; // Zero selects a sole instance; shared packages may require explicit dimensions.
    int steps = 8;
    uint32_t seed = 42;
    std::string pe_model; // Empty disables optional CPU prompt enhancement.
    PeOptions pe;
    // Diagnostics: raw FP32 inputs and a new trace directory, all optional.
    std::string latent, embeddings, trace;
    // Execution controls appended to preserve positional aggregate initialization of older callers.
    int threads = 4;
    int gpu_index = -1; // -1 selects ncnn's default Vulkan device.
    std::string text_device = "cpu";
    std::optional<RgbImage> input_image;
    float strength = .5f;
    bool text_down_vector = false; // Opt-in FP32 reduction candidate; independently exported 64/2048 buckets.
    // Informational img2img preprocessing identity. API callers normally use
    // "none" because their RGB buffer already has the requested dimensions.
    std::string input_resize = "none";
    int input_source_width = 0, input_source_height = 0;
    std::array<uint8_t, 3> input_alpha_background{255, 255, 255};
    std::array<uint8_t, 3> input_resize_background{0, 0, 0};
    std::string dit_weights = "auto"; // auto, device, host (RAM); Vulkan DiT only.
    // Extra headroom beyond estimated weight payload; a policy setting, not a
    // guarantee that the remaining activation/workspace allocations will fit.
    uint32_t gpu_reserve_mib = 512;
    // Opt-in FP32 Vulkan prepared-weight cache in RAM; 0 keeps streaming.
    // Charged weight bytes plus margin, not a whole-process memory ceiling.
    uint32_t dit_cache_mib = 0;
    uint32_t ram_reserve_mib = 3072;
    // Image text/DiT/VAE loading only; the separate prompt enhancer is unchanged.
    // default preserves the build setting. mapped may fall back to file reads.
    std::string model_loading = "default"; // default, stdio, mapped.
    // Optional preparation of one upcoming DiT block while the current block
    // computes. FP32 Vulkan with auto/host weights only; 0 disables admission.
    // Charged preparation budget, not a speed guarantee or process RAM limit.
    uint32_t dit_prefetch_mib = 0;
    // Vulkan DiT activation/workspace buffers only. Host-visible storage keeps
    // computation on the GPU. The existing RAM reserve also covers admission
    // of these buffers and prefetched weights, independently of cache use.
    std::string gpu_memory = "auto"; // auto, device, host.
    uint32_t gpu_spill_mib = 2048; // Cap on actual host-backed Vulkan allocation bytes.
    // At most three retries after a recognized allocation failure. The FP32
    // non-Flash attention path reduces query chunks 128 -> 64 -> 32 -> 16;
    // resolution and model mathematics stay fixed.
    uint32_t oom_retries = 3;
};

struct GenerationResult
{
    RgbImage image;
    std::string prompt; // Exact text consumed by the image text encoder.
    std::vector<uint32_t> token_ids;
    bool pe_eos = false;
    size_t pe_generated_tokens = 0;
    double elapsed_seconds = 0, vae_seconds = 0;
    uint64_t host_weight_requests = 0, device_weight_requests = 0;
    uint64_t unavailable_memory_queries = 0;
    uint64_t weight_cache_hits = 0, weight_cache_loads = 0;
    uint64_t weight_cache_peak_bytes = 0, weight_cache_peak_nets = 0;
    uint64_t weight_cache_evictions = 0, unavailable_host_memory_queries = 0;
    bool mapped_model_loading_requested = false; // Policy, not observed mapping success.
    // Resolved package/source metadata, available without tensor tracing.
    int model_schema = 0, text_bucket = 0, dit_text_tokens = 0;
    int source_width = 0, source_height = 0;
    int vulkan_gpu_index = -1; // -1 when no stage uses Vulkan.
    uint64_t prefetch_started = 0, prefetch_used = 0, prefetch_skipped = 0;
    uint64_t prefetch_peak_charged_bytes = 0;
    double prefetch_overlap_seconds = 0;
    uint64_t gpu_device_allocations = 0, gpu_host_allocations = 0, gpu_host_peak_bytes = 0;
    uint64_t gpu_memory_fallbacks = 0, gpu_allocation_failures = 0;
    // Some unified-memory devices expose only host-visible device-local RAM.
    uint64_t gpu_host_device_local_allocations = 0, gpu_host_non_device_local_allocations = 0;
    uint32_t memory_retries = 0;
    uint32_t attention_query_rows = 128; // Applied by the FP32 non-Flash attention path.
};

struct Progress
{
    std::string stage; // verify, pe-load, pe-prefill, pe-decode, text, denoise.
    int current = 0, total = 0;
    double seconds = 0;
};
using ProgressCallback = std::function<void(const Progress &)>;

struct VulkanDeviceInfo
{
    int index = -1;
    std::string name;
    bool fp16_storage = false, bf16_storage = false;
};

struct DiagnosticInfo
{
    bool vulkan_compiled = false;
    int default_gpu_index = -1;
    std::vector<VulkanDeviceInfo> vulkan_devices;
    bool model_config_loaded = false;
    int model_config_schema = 1;
    int packed_width = 0, packed_height = 0, text_bucket = 0, dit_text_tokens = 0;
    int text_layers = 0, dit_layers = 0;
    // A compiled Vulkan runtime can still lack a usable driver/device. Diagnosis
    // returns this reason and no devices; requesting Vulkan inference still fails.
    std::string vulkan_error;
};

// Verify every image runtime file without loading weights or starting Vulkan.
void verify_model(const std::string &directory);
void verify_pe_model(const std::string &directory);
// Reports compiled/runtime devices and optional model.cfg metadata. It does not
// verify a package or load model weights.
DiagnosticInfo diagnose(const std::string &model_directory = {});
// Owns each inference session and releases PE, text and DiT weights before the
// next stage. Returns pixels; callers choose their UI and image file format.
// Vulkan uses ncnn's process-global device context: serialize generate calls.
GenerationResult generate(const GenerationRequest &request, const ProgressCallback &progress = {});
} // namespace ernie

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
};

struct GenerationResult
{
    RgbImage image;
    std::string prompt; // Exact text consumed by the image text encoder.
    std::vector<uint32_t> token_ids;
    bool pe_eos = false;
    size_t pe_generated_tokens = 0;
    double elapsed_seconds = 0, vae_seconds = 0;
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

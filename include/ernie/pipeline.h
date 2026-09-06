// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
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
    int width = 0, height = 0; // Zero selects the model package's static size.
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
    std::optional<RgbImage> input_image; // Reserved until the F2 img2img runtime is connected.
    float strength = .5f;
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

// Verify every image runtime file without loading weights or starting Vulkan.
void verify_model(const std::string &directory);
void verify_pe_model(const std::string &directory);
// Owns each inference session and releases PE, text and DiT weights before the
// next stage. Returns pixels; callers choose their UI and image file format.
// Vulkan uses ncnn's process-global device context: serialize generate calls.
GenerationResult generate(const GenerationRequest &request, const ProgressCallback &progress = {});
} // namespace ernie

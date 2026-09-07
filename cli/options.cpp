// SPDX-License-Identifier: MIT
#include "options.h"
#include "prompt_file.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <stdexcept>
namespace fs = std::filesystem;
namespace ernie::cli
{
namespace
{
uint32_t integer(const std::string &value)
{
    size_t used = 0;
    const auto n = std::stoull(value, &used);
    if (used != value.size() || value.empty() || value[0] == '-' || n > UINT32_MAX)
        throw std::invalid_argument("Invalid integer");
    return uint32_t(n);
}
std::array<uint8_t, 3> color(const std::string &value)
{
    if (value.size() != 7 || value[0] != '#')
        throw std::invalid_argument("Background must be #RRGGBB");
    std::array<uint8_t, 3> result{};
    for (int i = 0; i < 3; ++i)
    {
        if (!std::isxdigit(static_cast<unsigned char>(value[1 + i * 2])) ||
            !std::isxdigit(static_cast<unsigned char>(value[2 + i * 2])))
            throw std::invalid_argument("Background must be #RRGGBB");
        size_t used = 0;
        const auto byte = std::stoul(value.substr(1 + i * 2, 2), &used, 16);
        if (used != 2)
            throw std::invalid_argument("Background must be #RRGGBB");
        result[i] = uint8_t(byte);
    }
    return result;
}
float number(const std::string &value)
{
    size_t used = 0;
    const float n = std::stof(value, &used);
    if (used != value.size() || !std::isfinite(n))
        throw std::invalid_argument("Invalid finite number");
    return n;
}
void output_extension(const fs::path &path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return char(std::tolower(c)); });
    if (extension != ".png" && extension != ".jpg" && extension != ".jpeg" && extension != ".bmp" &&
        extension != ".tga")
        throw std::invalid_argument("Output extension must be PNG, JPEG, BMP, or TGA");
}
} // namespace
const char *usage()
{
    return "ernie-image --model DIR (--prompt TEXT | --prompt-file UTF8.txt) --output NEW.{png|jpg|bmp|tga}\n"
           "            [--device cpu|vulkan] [--precision fp32|fp16|bf16]\n"
           "            [--width N --height N] [--seed N] [--steps N] [--threads N]\n"
           "            [--gpu N] [--text-device cpu]\n"
           "            [--dit-weights auto|device|host] [--gpu-reserve-mib N] (host uses RAM)\n"
           "            [--dit-cache-mib N] [--ram-reserve-mib N] (optional FP32 Vulkan RAM cache)\n"
           "            [--text-down-vector] (optional FP32 text reduction candidate)\n"
           "            [--vae-device cpu|vulkan] [--vae-convolution direct|sgemm]\n"
           "            [--pe-model DIR] [--pe-max-tokens N] [--pe-greedy]\n"
           "            [--pe-temperature N] [--pe-top-p N] [--pe-seed N]\n"
           "            [--metrics-json NEW.json] (allocation diagnostic build only; not a speed round)\n"
           "            [--latent FILE.f32] [--embeddings FILE.f32] [--trace-dir NEWDIR]\n"
           "            [--input IMAGE --strength 0..1 [--resize stretch|fit|crop] [--background #RRGGBB]]\n"
           "ernie-image (--model DIR | --pe-model DIR) --verify-model\n"
           "ernie-image [--model DIR] --diagnose\n"
           "Text-to-image: ernie-image --model model --prompt cat --output cat.png\n"
           "PE: ernie-image --model model --prompt cat --pe-model pe --pe-greedy --output cat.jpg\n"
           "Img2img: ernie-image --model model --input source.jpg --prompt style --width 512 --height 384 --resize fit --output result.png\n"
           "Resolution defaults to a sole model instance; shared packages require a supported width/height.\n"
           "Prompt files: UTF-8, optional BOM, at most 1 MiB; whitespace is preserved.\n"
           "PE is optional CPU FP32, with up to 2048 output tokens by default.\n"
           "PE sampling defaults: temperature 0.6, top-p 0.95, seed 42; --pe-greedy disables sampling.\n"
           "Reviewed model instances; FP32 text encoder, residuals, Euler master latent and\n"
           "FP32 VAE (CPU default). CFG=1. BF16 quality is experimental.\n";
}
Options parse_options(int argc, char **argv)
{
    Options out;
    auto &r = out.generation;
    std::set<std::string> seen;
    bool have_prompt = false, from_file = false, pe_option = false, background_option = false,
         strength_option = false, resize_option = false, gpu_option = false;
    fs::path prompt_file;
    for (int i = 1; i < argc; ++i)
    {
        const std::string flag(argv[i]);
        if (flag == "--help")
        {
            out.help = true;
            return out;
        }
        if (flag == "--prompt" || flag == "--prompt-file")
        {
            if (have_prompt)
                throw std::invalid_argument("Specify exactly one prompt source: --prompt or --prompt-file");
            have_prompt = true;
        }
        if (!seen.insert(flag).second)
            throw std::invalid_argument("Duplicate option: " + flag);
        if (flag == "--verify-model")
        {
            out.verify_only = true;
            continue;
        }
        if (flag == "--diagnose")
        {
            out.diagnose_only = true;
            continue;
        }
        if (flag == "--pe-greedy")
        {
            r.pe.greedy = true;
            pe_option = true;
            continue;
        }
        if (flag == "--text-down-vector")
        {
            r.text_down_vector = true;
            continue;
        }
        if (++i == argc)
            throw std::invalid_argument("Missing value for " + flag);
        const std::string value(argv[i]);
        if (flag == "--model")
            r.model = value;
        else if (flag == "--output")
            out.output = value;
        else if (flag == "--metrics-json")
            out.metrics_json = value;
        else if (flag == "--input")
            out.input = value;
        else if (flag == "--prompt")
            r.prompt = value;
        else if (flag == "--prompt-file")
        {
            prompt_file = value;
            from_file = true;
        }
        else if (flag == "--device")
            r.device = value;
        else if (flag == "--precision")
            r.precision = value;
        else if (flag == "--vae-device")
            r.vae_device = value;
        else if (flag == "--vae-convolution")
            r.vae_convolution = value;
        else if (flag == "--text-device")
            r.text_device = value;
        else if (flag == "--dit-weights")
            r.dit_weights = value;
        else if (flag == "--gpu-reserve-mib")
            r.gpu_reserve_mib = integer(value);
        else if (flag == "--dit-cache-mib")
            r.dit_cache_mib = integer(value);
        else if (flag == "--ram-reserve-mib")
            r.ram_reserve_mib = integer(value);
        else if (flag == "--background")
        {
            out.background = color(value);
            background_option = true;
            out.background_explicit = true;
        }
        else if (flag == "--resize")
        {
            out.resize = value;
            resize_option = true;
        }
        else if (flag == "--strength")
        {
            r.strength = number(value);
            strength_option = true;
        }
        else if (flag == "--latent")
            r.latent = value;
        else if (flag == "--embeddings")
            r.embeddings = value;
        else if (flag == "--trace-dir")
            r.trace = value;
        else if (flag == "--pe-model")
            r.pe_model = value;
        else if (flag == "--seed")
            r.seed = integer(value);
        else if (flag == "--steps")
        {
            const auto n = integer(value);
            if (n < 1 || n > 1000)
                throw std::invalid_argument("Steps must be in [1,1000]");
            r.steps = int(n);
        }
        else if (flag == "--threads")
        {
            const auto n = integer(value);
            if (n < 1 || n > 256)
                throw std::invalid_argument("Threads must be in [1,256]");
            r.threads = int(n);
        }
        else if (flag == "--gpu")
        {
            const auto n = integer(value);
            if (n > 63)
                throw std::invalid_argument("GPU index must be in [0,63]");
            r.gpu_index = int(n);
            gpu_option = true;
        }
        else if (flag == "--width" || flag == "--height")
        {
            const auto n = integer(value);
            if (n < 16 || n > 2048 || n % 16)
                throw std::invalid_argument("Width and height must be multiples of 16 in [16,2048]");
            (flag == "--width" ? r.width : r.height) = int(n);
        }
        else if (flag == "--pe-max-tokens")
        {
            const auto n = integer(value);
            if (n < 1 || n > 2048)
                throw std::invalid_argument("PE max tokens must be in [1,2048]");
            r.pe.max_tokens = int(n);
            pe_option = true;
        }
        else if (flag == "--pe-temperature")
        {
            r.pe.temperature = number(value);
            pe_option = true;
        }
        else if (flag == "--pe-top-p")
        {
            r.pe.top_p = number(value);
            pe_option = true;
        }
        else if (flag == "--pe-seed")
        {
            r.pe.seed = integer(value);
            pe_option = true;
        }
        else
            throw std::invalid_argument("Unknown argument: " + flag);
    }
    if (seen.count("--metrics-json"))
    {
#ifndef ERNIE_CLI_ALLOCATION_METRICS
        throw std::invalid_argument("This build has no allocation instrumentation; --metrics-json requires an ON build");
#else
        if (out.metrics_json.empty())throw std::invalid_argument("Allocation report path is empty");
        if (out.verify_only || out.diagnose_only)throw std::invalid_argument("--metrics-json requires generation");
        if (fs::exists(fs::symlink_status(out.metrics_json)))throw std::invalid_argument("Use a new allocation report path");
        if (!out.output.empty() && fs::absolute(out.metrics_json).lexically_normal()==fs::absolute(out.output).lexically_normal())
            throw std::invalid_argument("Allocation report and image paths must differ");
#endif
    }
    if (bool(r.width) != bool(r.height))
        throw std::invalid_argument("Specify width and height together");
    if (out.diagnose_only)
    {
        const size_t allowed = r.model.empty() ? 1 : 2;
        if (out.verify_only || seen.size() != allowed)
            throw std::invalid_argument("--diagnose accepts only an optional --model DIR");
        return out;
    }
    if (from_file)
        r.prompt = read_prompt(prompt_file);
    if (pe_option && r.pe_model.empty())
        throw std::invalid_argument("PE options require --pe-model");
    if (r.pe.temperature <= 0 || r.pe.top_p <= 0 || r.pe.top_p > 1)
        throw std::invalid_argument("PE temperature must be positive and top-p in (0,1]");
    if (!r.pe_model.empty() && !r.embeddings.empty())
        throw std::invalid_argument("PE and precomputed embeddings cannot be combined");
    if (r.text_down_vector && !r.embeddings.empty())
        throw std::invalid_argument("Vector text reduction requires native text encoding");
    if ((background_option || strength_option || resize_option) && out.input.empty())
        throw std::invalid_argument("--background, --strength and --resize require --input");
    if (!out.resize.empty() && out.resize != "stretch" && out.resize != "fit" && out.resize != "crop")
        throw std::invalid_argument("Resize must be stretch, fit, or crop");
    if (r.strength < 0 || r.strength > 1)
        throw std::invalid_argument("Strength must be in [0,1]");
    if (gpu_option && r.device != "vulkan" && r.vae_device != "vulkan")
        throw std::invalid_argument("--gpu requires a Vulkan generation or VAE device");
    if (r.text_device != "cpu")
        throw std::invalid_argument("Only --text-device cpu is currently supported");
    if (!out.verify_only && !out.output.empty())
        output_extension(out.output);
    const bool prompt_optional = !out.input.empty() && r.strength == 0.f;
    if (prompt_optional && (have_prompt || !r.pe_model.empty() || !r.embeddings.empty() || r.text_down_vector))
        throw std::invalid_argument("Strength-zero img2img does not consume prompt, PE, embeddings, or text reduction");
    if (!out.resize.empty() && (!r.width || !r.height))
        throw std::invalid_argument("--resize requires explicit --width and --height");
    if ((r.model.empty() && !(out.verify_only && !r.pe_model.empty())) ||
        (!out.verify_only && (out.output.empty() || (!have_prompt && !prompt_optional) || fs::exists(out.output))) ||
        (!r.trace.empty() && fs::exists(r.trace)) || (r.device != "cpu" && r.device != "vulkan") ||
        (r.precision != "fp32" && r.precision != "fp16" && r.precision != "bf16") ||
        (r.device == "cpu" && r.precision != "fp32") || (r.vae_device != "cpu" && r.vae_device != "vulkan") ||
        (r.vae_convolution != "sgemm" && r.vae_convolution != "direct"))
        throw std::invalid_argument("Invalid request; see --help and use new output paths");
    return out;
}
} // namespace ernie::cli

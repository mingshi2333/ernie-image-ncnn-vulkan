// SPDX-License-Identifier: MIT
#include "options.h"
#include "prompt_file.h"
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
float number(const std::string &value)
{
    size_t used = 0;
    const float n = std::stof(value, &used);
    if (used != value.size() || !std::isfinite(n))
        throw std::invalid_argument("Invalid finite number");
    return n;
}
} // namespace
const char *usage()
{
    return "ernie-image --model DIR (--prompt TEXT | --prompt-file UTF8.txt) --output NEW.png\n"
           "            [--device cpu|vulkan] [--precision fp32|fp16|bf16]\n"
           "            [--width N --height N] [--seed N] [--steps N]\n"
           "            [--vae-device cpu|vulkan] [--vae-convolution direct|sgemm]\n"
           "            [--pe-model DIR] [--pe-max-tokens N] [--pe-greedy]\n"
           "            [--pe-temperature N] [--pe-top-p N] [--pe-seed N]\n"
           "            [--latent FILE.f32] [--embeddings FILE.f32] [--trace-dir NEWDIR]\n"
           "ernie-image (--model DIR | --pe-model DIR) --verify-model\n"
           "Resolution defaults to the model bucket; explicit dimensions must match it.\n"
           "Prompt files: UTF-8, optional BOM, at most 1 MiB; whitespace is preserved.\n"
           "PE is optional CPU FP32, with up to 2048 output tokens by default.\n"
           "PE sampling defaults: temperature 0.6, top-p 0.95, seed 42; --pe-greedy disables sampling.\n"
           "Experimental static model bucket; FP32 text encoder, residuals, Euler master latent and\n"
           "FP32 VAE (CPU default). CFG=1. BF16 quality is experimental.\n";
}
Options parse_options(int argc, char **argv)
{
    Options out;
    auto &r = out.generation;
    std::set<std::string> seen;
    bool have_prompt = false, from_file = false, pe_option = false;
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
        if (flag == "--pe-greedy")
        {
            r.pe.greedy = true;
            pe_option = true;
            continue;
        }
        if (++i == argc)
            throw std::invalid_argument("Missing value for " + flag);
        const std::string value(argv[i]);
        if (flag == "--model")
            r.model = value;
        else if (flag == "--output")
            out.output = value;
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
    if (bool(r.width) != bool(r.height))
        throw std::invalid_argument("Specify width and height together");
    if (from_file)
        r.prompt = read_prompt(prompt_file);
    if (pe_option && r.pe_model.empty())
        throw std::invalid_argument("PE options require --pe-model");
    if (r.pe.temperature <= 0 || r.pe.top_p <= 0 || r.pe.top_p > 1)
        throw std::invalid_argument("PE temperature must be positive and top-p in (0,1]");
    if (!r.pe_model.empty() && !r.embeddings.empty())
        throw std::invalid_argument("PE and precomputed embeddings cannot be combined");
    if ((r.model.empty() && !(out.verify_only && !r.pe_model.empty())) ||
        (!out.verify_only && (out.output.empty() || !have_prompt || fs::exists(out.output))) ||
        (!r.trace.empty() && fs::exists(r.trace)) || (r.device != "cpu" && r.device != "vulkan") ||
        (r.precision != "fp32" && r.precision != "fp16" && r.precision != "bf16") ||
        (r.device == "cpu" && r.precision != "fp32") || (r.vae_device != "cpu" && r.vae_device != "vulkan") ||
        (r.vae_convolution != "sgemm" && r.vae_convolution != "direct"))
        throw std::invalid_argument("Invalid request; see --help and use new output paths");
    return out;
}
} // namespace ernie::cli

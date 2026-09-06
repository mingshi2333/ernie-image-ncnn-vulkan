// SPDX-License-Identifier: MIT
#include "prompt_enhancer.h"
#include "ernie_gelu.h"
#include "pe_session.h"
#include "text_encoder.h"
#include "tokenizer.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace ernie
{
namespace
{
void check(int rc, const char *action)
{
    if (rc)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(rc));
}
void options_check(const PeOptions &options)
{
    if (options.max_tokens < 1 || options.max_tokens > 2048 || !std::isfinite(options.temperature) ||
        options.temperature <= 0 || !std::isfinite(options.top_p) || options.top_p <= 0 || options.top_p > 1)
        throw std::invalid_argument(
            "PE requires max tokens 1..2048, positive temperature and top-p in (0,1]");
}
} // namespace

uint32_t sample_pe_token(const ncnn::Mat &logits, const PeOptions &options, std::mt19937 &random)
{
    options_check(options);
    if (logits.empty() || logits.elempack != 1 || logits.elemsize != 4u || logits.h != 1 || logits.c != 1 ||
        logits.d != 1)
        throw std::invalid_argument("PE logits must be an FP32 vector");
    const float *values = logits;
    for (int i = 0; i < logits.w; ++i)
        if (!std::isfinite(values[i]))
            throw std::runtime_error("Non-finite PE logits");
    const int best = int(std::max_element(values, values + logits.w) - values);
    if (options.greedy)
        return uint32_t(best);
    std::vector<uint32_t> order(logits.w);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return values[a] > values[b]; });
    std::vector<double> weights(order.size());
    double total = 0;
    for (size_t i = 0; i < order.size(); ++i)
    {
        weights[i] = std::exp((double(values[order[i]]) - values[best]) / options.temperature);
        total += weights[i];
    }
    double cumulative = 0;
    size_t retained = 0;
    do
    {
        cumulative += weights[retained++];
    } while (retained < weights.size() && cumulative < options.top_p * total);
    weights.resize(retained);
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return order[distribution(random)];
}

PeResult enhance_prompt(const std::string &model, const std::string &prompt, int width, int height,
                        const PeOptions &options, const PeProgress &progress, const PeLogits &observe)
{
    options_check(options);
    const std::filesystem::path root(model);
    verify_package(model);
    Tokenizer tokenizer((root / "tokenizer").string());
    PeResult result;
    result.input_ids = tokenizer.encode_exact(pe_chat_prompt(prompt, width, height));
    if (result.input_ids.empty() || result.input_ids.size() > 2048)
        throw std::invalid_argument("PE chat input exceeds 2048 tokens");
    float frequencies[64];
    std::ifstream rope(root / "rope-inv-freq.f32", std::ios::binary);
    if (std::filesystem::file_size(root / "rope-inv-freq.f32") != sizeof(frequencies) ||
        !rope.read(reinterpret_cast<char *>(frequencies), sizeof(frequencies)))
        throw std::runtime_error("Invalid PE YaRN table");
    for (float f : frequencies)
        if (!std::isfinite(f) || f <= 0)
            throw std::runtime_error("Invalid PE frequency");
    ncnn::Option cpu;
    cpu.num_threads = 4;
    cpu.use_vulkan_compute = false;
    cpu.use_fp16_storage = cpu.use_fp16_packed = cpu.use_fp16_arithmetic = false;
    cpu.use_bf16_storage = cpu.use_bf16_packed = false;
    std::vector<std::unique_ptr<ncnn::Net>> blocks;
    std::vector<const ncnn::Net *> pointers;
    for (int i = 0; i < 26; ++i)
    {
        auto net = std::make_unique<ncnn::Net>();
        net->opt = cpu;
        load_pe_block(*net,
                      (root / (std::string("block-") + (i < 10 ? "0" : "") + std::to_string(i))).string());
        pointers.push_back(net.get());
        blocks.push_back(std::move(net));
        if (progress)
            progress("load", i + 1, 26);
    }
    ncnn::Net head;
    head.opt = cpu;
    check(register_layers(head), "Register PE head");
    check(head.load_param((root / "head.ncnn.param").string().c_str()), "Load PE head graph");
    check(head.load_model((root / "head.ncnn.bin").string().c_str()), "Load PE head weights");
    PeSession session(pointers, int(result.input_ids.size()) + options.max_tokens);
    auto advance = [&](uint32_t token)
    {
        const auto embedded = text_embeddings((root / "embeddings.bf16").string(), {token}, 1);
        ncnn::Mat cos(128, 1), sin(128, 1);
        if (cos.empty() || sin.empty())
            throw std::bad_alloc();
        for (int i = 0; i < 64; ++i)
        {
            const float phase = float(session.position()) * frequencies[i];
            cos[i] = cos[i + 64] = std::cos(phase);
            sin[i] = sin[i + 64] = std::sin(phase);
        }
        return session.step(embedded, cos, sin);
    };
    ncnn::Mat hidden;
    for (size_t i = 0; i < result.input_ids.size(); ++i)
    {
        hidden = advance(result.input_ids[i]);
        if (progress && ((i + 1) % 8 == 0 || i + 1 == result.input_ids.size()))
            progress("prefill", int(i + 1), int(result.input_ids.size()));
    }
    std::mt19937 random(options.seed);
    for (int i = 0; i < options.max_tokens; ++i)
    {
        auto ex = head.create_extractor();
        check(ex.input("in0", hidden), "Input PE head");
        ncnn::Mat logits;
        check(ex.extract("out0", logits), "Extract PE logits");
        if (logits.w != 131072 || logits.total() != 131072)
            throw std::runtime_error("PE vocabulary shape differs");
        if (observe)
            observe(i, logits);
        const uint32_t token = sample_pe_token(logits, options, random);
        result.generated_ids.push_back(token);
        if (progress && ((i + 1) % 8 == 0 || token == 2 || i + 1 == options.max_tokens))
            progress("decode", i + 1, options.max_tokens);
        if (token == 2)
        {
            result.eos = true;
            break;
        }
        if (i + 1 < options.max_tokens)
            hidden = advance(token);
    }
    result.cache_buffer_changes = session.cache_buffer_changes();
    result.text = trim_whitespace(tokenizer.decode(result.generated_ids));
    if (result.text.empty())
        throw std::runtime_error("PE produced an empty description");
    return result;
}
} // namespace ernie

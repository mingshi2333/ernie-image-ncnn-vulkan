// SPDX-License-Identifier: MIT
#include "prompt_enhancer.h"
#include "tokenizer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace fs = std::filesystem;
int main(int argc, char **argv)
{
    try
    {
        if (argc != 8)
            throw std::invalid_argument(
                "ernie-pe MODEL PROMPT_FILE WIDTH HEIGHT MAX_TOKENS NEW_OUTPUT greedy|sample|tokenize");
        const fs::path out(argv[6]);
        if (fs::exists(out))
            throw std::invalid_argument("Use new PE output directory");
        std::ifstream file(argv[2], std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot read PE prompt");
        const std::string prompt((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
        const std::string mode(argv[7]);
        if (mode != "greedy" && mode != "sample" && mode != "tokenize")
            throw std::invalid_argument("Invalid PE mode");
        fs::create_directories(out);
        if (mode == "tokenize")
        {
            ernie::Tokenizer tokenizer(argv[1]);
            const auto formatted = ernie::pe_chat_prompt(prompt, width, height);
            const auto ids = tokenizer.encode_exact(formatted);
            std::ofstream(out / "formatted.txt", std::ios::binary) << formatted;
            std::ofstream decoded(out / "decoded.txt", std::ios::binary);
            decoded << tokenizer.decode(ids);
            std::ofstream tokens(out / "input-ids.txt");
            for (auto id : ids)
                tokens << id << '\n';
            return 0;
        }
        ernie::PeOptions options;
        options.max_tokens = std::stoi(argv[5]);
        options.greedy = mode == "greedy";
        auto result = ernie::enhance_prompt(
            argv[1], prompt, width, height, options, [](const char *phase, int step, int total)
            { std::cout << "PE " << phase << ' ' << step << '/' << total << std::endl; },
            [&](int step, const ncnn::Mat &logits)
            {
                std::ofstream file(out / ("logits-" + std::to_string(step) + ".f32"), std::ios::binary);
                file.write(static_cast<const char *>(logits.data), size_t(logits.w) * 4);
                if (!file)
                    throw std::runtime_error("Cannot write PE logits");
            });
        std::ofstream(out / "enhanced.txt", std::ios::binary) << result.text;
        std::ofstream ids(out / "input-ids.txt"), generated(out / "generated-ids.txt");
        for (auto id : result.input_ids)
            ids << id << '\n';
        for (auto id : result.generated_ids)
            generated << id << '\n';
        std::ofstream(out / "status.txt")
            << "eos " << result.eos << "\ncache_buffer_changes " << result.cache_buffer_changes << '\n';
        std::cout << "Enhanced: " << result.text << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

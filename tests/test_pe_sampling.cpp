// SPDX-License-Identifier: MIT
#include "prompt_enhancer.h"
#include "tokenizer.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}
int main()
{
    try
    {
        require(ernie::trim_whitespace(u8"\u3000\x1c\t猫 \n犬\u00a0\r\n") == u8"猫 \n犬",
                "PE Unicode trim differs from Python");
        require(ernie::trim_whitespace("\x1f\t\n").empty(), "Whitespace-only PE output differs");
        ernie::PeOptions options;
        ncnn::Mat logits(3);
        logits[0] = -1;
        logits[1] = 4;
        logits[2] = 4;
        std::mt19937 first(42), second(42);
        options.greedy = true;
        require(ernie::sample_pe_token(logits, options, first) == 1, "Greedy tie must choose first token");
        options.greedy = false;
        options.temperature = 1;
        options.top_p = .6f;
        ncnn::Mat distribution_logits(3);
        distribution_logits[0] = std::log(.7f);
        distribution_logits[1] = std::log(.2f);
        distribution_logits[2] = std::log(.1f);
        for (int i = 0; i < 32; ++i)
            require(ernie::sample_pe_token(distribution_logits, options, first) == 0,
                    "Top-p 0.6 must retain only the 0.7 token");
        options.top_p = 1;
        first.seed(20260906);
        std::array<int, 3> counts{};
        constexpr int draws = 100000;
        for (int i = 0; i < draws; ++i)
            ++counts.at(ernie::sample_pe_token(distribution_logits, options, first));
        for (int i = 0; i < 3; ++i)
        {
            constexpr std::array<double, 3> expected{.7, .2, .1};
            require(std::abs(double(counts[i]) / draws - expected[i]) <= .01,
                    "Native sampling frequency exceeds the declared tolerance");
        }
        first.seed(42);
        second.seed(42);
        for (int i = 0; i < 32; ++i)
            require(ernie::sample_pe_token(logits, options, first) ==
                        ernie::sample_pe_token(logits, options, second),
                    "Native sampling must be reproducible within one implementation");
        for (float bad : {0.f, -1.f, INFINITY, NAN})
        {
            options.temperature = bad;
            bool rejected = false;
            try
            {
                ernie::sample_pe_token(logits, options, first);
            }
            catch (const std::invalid_argument &)
            {
                rejected = true;
            }
            require(rejected, "Invalid temperature accepted");
        }
        options.temperature = 1;
        for (float bad : {0.f, -1.f, 1.01f, INFINITY, NAN})
        {
            options.top_p = bad;
            bool rejected_top_p = false;
            try
            {
                ernie::sample_pe_token(distribution_logits, options, first);
            }
            catch (const std::invalid_argument &)
            {
                rejected_top_p = true;
            }
            require(rejected_top_p, "Invalid top-p accepted");
        }
        options.top_p = 1;
        for (int bad : {0, -1, 2049})
        {
            options.max_tokens = bad;
            bool rejected_max_tokens = false;
            try
            {
                ernie::sample_pe_token(distribution_logits, options, first);
            }
            catch (const std::invalid_argument &)
            {
                rejected_max_tokens = true;
            }
            require(rejected_max_tokens, "Invalid max_tokens accepted");
        }
        options.max_tokens = 256;
        ncnn::Mat empty;
        bool rejected_empty = false;
        try
        {
            ernie::sample_pe_token(empty, options, first);
        }
        catch (const std::invalid_argument &)
        {
            rejected_empty = true;
        }
        require(rejected_empty, "Empty logits accepted");
        logits[0] = NAN;
        bool rejected = false;
        try
        {
            ernie::sample_pe_token(logits, options, first);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        require(rejected, "Non-finite logits accepted");
        std::cout << "PE greedy, 100000-draw distribution, nucleus, bounds and finite-value contracts passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

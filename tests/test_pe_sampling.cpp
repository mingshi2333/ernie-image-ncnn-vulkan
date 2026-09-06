// SPDX-License-Identifier: MIT
#include "prompt_enhancer.h"
#include "tokenizer.h"
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
        options.top_p = .1f;
        for (int i = 0; i < 32; ++i)
            require(ernie::sample_pe_token(logits, options, first) == 1,
                    "Nucleus must exclude tokens outside top-p");
        first.seed(42);
        options.top_p = 1;
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
        std::cout << "PE greedy, nucleus, reproducibility and finite-value contracts passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

// SPDX-License-Identifier: MIT
#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv)
{
    if (argc < 3) { std::cerr << "Usage: ernie-tokenize TOKENIZER_DIRECTORY UTF8_PROMPT_FILE...\n"; return 2; }
    try
    {
        ernie::Tokenizer tokenizer(argv[1]);
        for (int i = 2; i < argc; ++i)
        {
            std::ifstream input(argv[i], std::ios::binary);
            if (!input) throw std::runtime_error("Cannot open prompt file");
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            const auto ids = tokenizer.encode(text);
            std::cout << "[";
            for (size_t j = 0; j < ids.size(); ++j) std::cout << (j ? "," : "") << ids[j];
            std::cout << "]\n";
        }
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

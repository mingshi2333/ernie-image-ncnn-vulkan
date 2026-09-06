// SPDX-License-Identifier: MIT
#include "prompt_file.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main()
{
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("ernie-prompt-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("Temporary directory exists");
        const auto file = directory / "prompt.txt";
        const std::string expected = u8" \t猫\r\n犬 \n";
        std::ofstream(file, std::ios::binary) << "\xef\xbb\xbf" << expected;
        if (ernie::cli::read_prompt(file) != expected)
            throw std::runtime_error("BOM/whitespace handling differs");
        for (const auto &invalid :
             {std::string("\xc0\xaf"), std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"),
              std::string("\xe4\xb8"), std::string("a\0b", 3)})
        {
            std::ofstream(file, std::ios::binary).write(invalid.data(), invalid.size());
            bool rejected = false;
            try
            {
                ernie::cli::read_prompt(file);
            }
            catch (const std::invalid_argument &)
            {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Invalid UTF-8 or NUL accepted");
        }
        std::filesystem::remove_all(directory);
        std::cout << "Prompt BOM, exact whitespace and UTF-8 boundaries passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}

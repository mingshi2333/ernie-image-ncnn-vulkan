// SPDX-License-Identifier: MIT
#include <windows.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

int ernie_cli_main(int argc, char** argv);

// Windows supplies UTF-16 arguments. Convert explicitly so prompts and public
// UTF-8 path strings do not depend on the machine's ANSI code page.
int wmain(int argc, wchar_t** argv)
{
    try
    {
        std::vector<std::string> arguments;
        arguments.reserve(argc);
        for (int i = 0; i < argc; ++i)
        {
            const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1,
                                                 nullptr, 0, nullptr, nullptr);
            if (!size) throw std::system_error(GetLastError(), std::system_category(), "Invalid command line");
            std::string value(size, '\0');
            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1,
                                    value.data(), size, nullptr, nullptr) != size)
                throw std::system_error(GetLastError(), std::system_category(), "Cannot decode command line");
            value.pop_back();
            arguments.push_back(std::move(value));
        }
        std::vector<char*> pointers;
        pointers.reserve(arguments.size() + 1);
        for (auto& value : arguments) pointers.push_back(value.data());
        pointers.push_back(nullptr);
        return ernie_cli_main(argc, pointers.data());
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

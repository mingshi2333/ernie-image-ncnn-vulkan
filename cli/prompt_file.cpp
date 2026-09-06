// SPDX-License-Identifier: MIT
#include "prompt_file.h"
#include <fstream>
#include <stdexcept>
namespace fs = std::filesystem;
namespace ernie::cli
{
std::string read_prompt(const fs::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file || !fs::is_regular_file(path))
        throw std::invalid_argument("Cannot open prompt file: " + path.string());
    constexpr size_t limit = 1024 * 1024;
    if (fs::file_size(path) > limit)
        throw std::invalid_argument("Prompt file exceeds 1 MiB");
    std::string text(limit + 1, '\0');
    file.read(text.data(), text.size());
    if (file.bad())
        throw std::runtime_error("Cannot read prompt file: " + path.string());
    text.resize(size_t(file.gcount()));
    if (text.size() > limit)
        throw std::invalid_argument("Prompt file exceeds 1 MiB");
    // Treat a UTF-8 file signature as encoding metadata; preserve all other
    // bytes, including leading/trailing spaces and CRLF line endings.
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
        text.erase(0, 3);
    for (size_t i = 0; i < text.size();)
    {
        const auto first = static_cast<unsigned char>(text[i++]);
        if (first == 0)
            throw std::invalid_argument("Prompt file contains NUL; require UTF-8 text");
        if (first < 0x80)
            continue;
        const int count = first >= 0xc2 && first <= 0xdf   ? 1
                          : first >= 0xe0 && first <= 0xef ? 2
                          : first >= 0xf0 && first <= 0xf4 ? 3
                                                           : -1;
        if (count < 0 || i + count > text.size())
            throw std::invalid_argument("Prompt file is not valid UTF-8");
        unsigned code = first & ((1u << (6 - count)) - 1);
        for (int j = 0; j < count; ++j)
        {
            const auto byte = static_cast<unsigned char>(text[i++]);
            if ((byte & 0xc0) != 0x80)
                throw std::invalid_argument("Prompt file is not valid UTF-8");
            code = (code << 6) | (byte & 0x3f);
        }
        if (code < (count == 1   ? 0x80u
                    : count == 2 ? 0x800u
                                 : 0x10000u) ||
            code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
            throw std::invalid_argument("Prompt file is not valid UTF-8");
    }
    return text;
}

} // namespace ernie::cli

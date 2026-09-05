// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ernie {
// Native C++ API, backed by pinned official Tokenizers through a C ABI.
// No Python interpreter, network access, or model weights are used at encoding.
class Tokenizer
{
public:
    explicit Tokenizer(const std::string& model_directory);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    std::vector<uint32_t> encode(const std::string& prompt) const;
private:
    void* handle_ = nullptr;
    size_t maximum_ = 0;
};
}

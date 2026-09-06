// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ernie
{
// Check every inference file before model loading, using its SHA256 manifest.
void verify_package(const std::string &model_directory);
// Matches Python str.strip(), including its four ASCII information separators.
std::string trim_whitespace(const std::string &text);
std::string pe_chat_prompt(const std::string &prompt, unsigned width, unsigned height);
// Native C++ API, backed by pinned official Tokenizers through a C ABI.
// No Python interpreter, network access, or model weights are used at encoding.
class Tokenizer
{
  public:
    explicit Tokenizer(const std::string &model_directory);
    ~Tokenizer();
    Tokenizer(const Tokenizer &) = delete;
    Tokenizer &operator=(const Tokenizer &) = delete;
    std::vector<uint32_t> encode(const std::string &prompt) const;
    // PE must reject over-capacity input rather than truncate a chat message.
    std::vector<uint32_t> encode_exact(const std::string &prompt) const;
    std::string decode(const std::vector<uint32_t> &ids) const;

  private:
    void *handle_ = nullptr;
    size_t maximum_ = 0;
};
} // namespace ernie

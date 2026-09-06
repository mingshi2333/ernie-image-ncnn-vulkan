// SPDX-License-Identifier: MIT
#include "tokenizer.h"
#include <stdexcept>

extern "C"
{
    int ernie_package_verify(const unsigned char *, size_t, unsigned char *, size_t);
    void *ernie_tok_create(const unsigned char *, size_t, unsigned char *, size_t);
    void *ernie_tok_create_files(const unsigned char *, size_t, const unsigned char *, size_t, unsigned char *, size_t);
    size_t ernie_tok_maximum(const void *);
    int ernie_tok_encode(const void *, const unsigned char *, size_t, uint32_t *, size_t, size_t *,
                         unsigned char *, size_t);
    int ernie_tok_encode_exact(const void *, const unsigned char *, size_t, uint32_t *, size_t, size_t *,
                               unsigned char *, size_t);
    int ernie_tok_decode(const void *, const uint32_t *, size_t, unsigned char *, size_t, size_t *,
                         unsigned char *, size_t);
    int ernie_pe_prompt(const unsigned char *, size_t, unsigned, unsigned, unsigned char *, size_t, size_t *,
                        unsigned char *, size_t);
    int ernie_trim_text(const unsigned char *, size_t, unsigned char *, size_t, size_t *, unsigned char *,
                        size_t);
    void ernie_tok_destroy(void *);
}
namespace ernie
{
void verify_package(const std::string &directory)
{
    unsigned char error[1024] = {};
    if (ernie_package_verify(reinterpret_cast<const unsigned char *>(directory.data()), directory.size(),
                             error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
}
Tokenizer::Tokenizer(const std::string &directory)
{
    unsigned char error[512] = {};
    handle_ = ernie_tok_create(reinterpret_cast<const unsigned char *>(directory.data()), directory.size(),
                               error, sizeof(error));
    if (!handle_)
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    maximum_ = ernie_tok_maximum(handle_);
}
Tokenizer::Tokenizer(const std::string &json, const std::string &config)
{
    unsigned char error[1024] = {};
    handle_ = ernie_tok_create_files(reinterpret_cast<const unsigned char *>(json.data()), json.size(),
                                   reinterpret_cast<const unsigned char *>(config.data()), config.size(),
                                   error, sizeof(error));
    if (!handle_) throw std::runtime_error(reinterpret_cast<const char *>(error));
    maximum_ = ernie_tok_maximum(handle_);
}
Tokenizer::~Tokenizer()
{
    ernie_tok_destroy(handle_);
}
std::string trim_whitespace(const std::string &text)
{
    unsigned char error[512] = {};
    size_t length = 0;
    const auto *data = reinterpret_cast<const unsigned char *>(text.data());
    if (ernie_trim_text(data, text.size(), nullptr, 0, &length, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    std::string output(length, '\0');
    if (ernie_trim_text(data, text.size(), reinterpret_cast<unsigned char *>(output.data()), output.size(),
                        &length, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    return output;
}
std::string pe_chat_prompt(const std::string &prompt, unsigned width, unsigned height)
{
    unsigned char error[512] = {};
    size_t length = 0;
    const auto *data = reinterpret_cast<const unsigned char *>(prompt.data());
    if (ernie_pe_prompt(data, prompt.size(), width, height, nullptr, 0, &length, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    std::string output(length, '\0');
    if (ernie_pe_prompt(data, prompt.size(), width, height, reinterpret_cast<unsigned char *>(output.data()),
                        output.size(), &length, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    return output;
}
std::vector<uint32_t> Tokenizer::encode_exact(const std::string &prompt) const
{
    std::vector<uint32_t> output(maximum_);
    size_t written = 0;
    unsigned char error[512] = {};
    if (ernie_tok_encode_exact(handle_, reinterpret_cast<const unsigned char *>(prompt.data()), prompt.size(),
                               output.data(), output.size(), &written, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    if (written > output.size())
        throw std::runtime_error("Tokenizer output exceeds capacity");
    output.resize(written);
    return output;
}
std::string Tokenizer::decode(const std::vector<uint32_t> &ids) const
{
    unsigned char error[512] = {};
    size_t length = 0;
    if (ernie_tok_decode(handle_, ids.data(), ids.size(), nullptr, 0, &length, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    std::string output(length, '\0');
    if (ernie_tok_decode(handle_, ids.data(), ids.size(), reinterpret_cast<unsigned char *>(output.data()),
                         output.size(), &length, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    return output;
}
std::vector<uint32_t> Tokenizer::encode(const std::string &prompt) const
{
    std::vector<uint32_t> output(maximum_);
    size_t written = 0;
    unsigned char error[512] = {};
    if (ernie_tok_encode(handle_, reinterpret_cast<const unsigned char *>(prompt.data()), prompt.size(),
                         output.data(), output.size(), &written, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char *>(error));
    if (written > output.size())
        throw std::runtime_error("Tokenizer output exceeds capacity");
    output.resize(written);
    return output;
}
} // namespace ernie

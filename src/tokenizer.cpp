// SPDX-License-Identifier: MIT
#include "tokenizer.h"
#include <stdexcept>

extern "C" {
int ernie_package_verify(const unsigned char*, size_t, unsigned char*, size_t);
void* ernie_tok_create(const unsigned char*, size_t, unsigned char*, size_t);
size_t ernie_tok_maximum(const void*);
int ernie_tok_encode(const void*, const unsigned char*, size_t, uint32_t*, size_t, size_t*, unsigned char*, size_t);
void ernie_tok_destroy(void*);
}
namespace ernie {
void verify_package(const std::string& directory)
{
    unsigned char error[1024] = {};
    if (ernie_package_verify(reinterpret_cast<const unsigned char*>(directory.data()), directory.size(), error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char*>(error));
}
Tokenizer::Tokenizer(const std::string& directory)
{
    unsigned char error[512] = {};
    handle_ = ernie_tok_create(reinterpret_cast<const unsigned char*>(directory.data()), directory.size(), error, sizeof(error));
    if (!handle_) throw std::runtime_error(reinterpret_cast<const char*>(error));
    maximum_ = ernie_tok_maximum(handle_);
}
Tokenizer::~Tokenizer() { ernie_tok_destroy(handle_); }
std::vector<uint32_t> Tokenizer::encode(const std::string& prompt) const
{
    std::vector<uint32_t> output(maximum_);
    size_t written = 0;
    unsigned char error[512] = {};
    if (ernie_tok_encode(handle_, reinterpret_cast<const unsigned char*>(prompt.data()), prompt.size(),
                         output.data(), output.size(), &written, error, sizeof(error)))
        throw std::runtime_error(reinterpret_cast<const char*>(error));
    if (written > output.size()) throw std::runtime_error("Tokenizer output exceeds capacity");
    output.resize(written);
    return output;
}
}

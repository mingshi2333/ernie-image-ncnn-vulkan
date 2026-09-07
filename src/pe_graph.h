// SPDX-License-Identifier: MIT
#pragma once
#include <string>
namespace ernie
{
// Internal exact-byte allowlist; only four sequence fields may change.
std::string pe_chunk_graph(const std::string &canonical);
} // namespace ernie

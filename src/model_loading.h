// SPDX-License-Identifier: MIT
#pragma once
#include <stdexcept>
#include <string>

namespace ernie {
// A request for ncnn's mapping loader, which can fall back to ordinary reads.
// Keep the existing build default while allowing both paths in one binary.
inline bool request_mapped_model_loading(const std::string& mode, bool build_default)
{
    if (mode == "default") return build_default;
    if (mode == "stdio") return false;
    if (mode == "mapped") return true;
    throw std::invalid_argument("Model loading must be default, stdio, or mapped");
}
} // namespace ernie

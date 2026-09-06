// SPDX-License-Identifier: MIT
#pragma once
#include "model_config.h"
#include <string>
namespace ernie
{
// Full-hash-reviewed graph text instantiation; no file writes or weight copies.
// Targets remain restricted to independently exported static configurations.
std::string shape_graph_sha256(const std::string &bytes);
std::string instantiate_shape_graph(const std::string &kind, const std::string &graph,
                                    const ModelConfig &source, const ModelConfig &target);
}

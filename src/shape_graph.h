// SPDX-License-Identifier: MIT
#pragma once
#include "model_config.h"
#include <string>
namespace ernie
{
// Offline candidate only: no file writes, weights, pipeline or schema-3 runtime acceptance.
std::string shape_graph_sha256(const std::string &bytes);
std::string instantiate_shape_graph(const std::string &kind, const std::string &graph,
                                    const ModelConfig &source, const ModelConfig &target);
}

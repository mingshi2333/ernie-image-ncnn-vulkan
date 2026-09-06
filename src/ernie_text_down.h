// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"
#include <string>
namespace ernie
{
// CPU FP32 only. Full graph token comparison precedes the single down replacement.
std::string vector_text_down_graph(const std::string& graph, int bucket);
// Validate the complete reviewed block's raw norm and tagged Gemm weight stream.
void validate_text_down_weights(const std::string& path);
int register_text_down(ncnn::Net& net);
}

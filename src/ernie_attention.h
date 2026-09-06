// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"
#include <cstdint>
namespace ernie {
// Disabling the workspace bound is for controlled diagnostics only.
int register_attention(ncnn::Net& net, bool bounded_workspace = true);
// Only call for Nets registered above. Includes internal chunk completions;
// the caller's final submission is counted separately.
uint64_t attention_internal_submissions(const ncnn::Net& net);
}

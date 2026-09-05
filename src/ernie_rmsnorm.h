// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"
namespace ernie {
// Keep RMSNorm square/reduction/normalization in FP32 even with FP16/BF16 storage.
int register_rmsnorm(ncnn::Net& net);
// ERNIE final non-affine LayerNorm has the same FP16 square overflow risk.
int register_layernorm(ncnn::Net& net);
}

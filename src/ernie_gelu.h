// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"

namespace ernie {
// Registers erf-form GELU and the FP32-intermediate RMSNorm runtime override.
int register_layers(ncnn::Net& net);
}

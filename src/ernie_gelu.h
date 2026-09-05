// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"

namespace ernie {
// Registers the erf-form GELU used by ERNIE. It has no learned parameters.
int register_layers(ncnn::Net& net);
}

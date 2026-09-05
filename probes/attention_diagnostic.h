// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"

struct AttentionDiagnostic
{
    int calls = 0;
    int flash_calls = 0;
    int cooperative_calls = 0;
    int tokens = 0;
    int head_dim = 0;
};

#if NCNN_VULKAN
int register_attention_diagnostic(ncnn::Net& net, AttentionDiagnostic& data);
#endif

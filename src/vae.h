// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"
#include <string>
namespace ernie
{
// The caller owns the Vulkan context, if selected. CPU FP32/direct is the
// validated default; no VAE weights survive the call.
ncnn::Mat decode_vae(const std::string &directory, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &device, const std::string &convolution);
} // namespace ernie

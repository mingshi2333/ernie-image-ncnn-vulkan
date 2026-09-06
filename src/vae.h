// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "net.h"
#include <string>
namespace ernie
{
// The caller owns the Vulkan context, if selected. CPU FP32/direct is the
// validated default; no VAE weights survive the call.
ncnn::Mat decode_vae(const ComponentFiles &files, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &device, const std::string &convolution, int gpu_index = -1);
ncnn::Mat decode_vae(const std::string &directory, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &device, const std::string &convolution, int gpu_index = -1);
} // namespace ernie

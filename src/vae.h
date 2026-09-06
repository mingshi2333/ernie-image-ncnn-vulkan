// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "net.h"
#include <string>
#include <vector>
namespace ernie
{
struct VaeStats
{
    struct Detail { std::string boundary,status; double seconds=0; };
    std::vector<Detail> details;
};
// The caller owns the Vulkan context, if selected. CPU FP32/direct is the
// validated default; no VAE weights survive the call.
ncnn::Mat decode_vae(const ComponentFiles &files, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &device, const std::string &convolution, int gpu_index = -1,
                     VaeStats *stats = nullptr);
ncnn::Mat decode_vae(const std::string &directory, const ncnn::Mat &unpacked, const ncnn::Option &cpu,
                     const std::string &device, const std::string &convolution, int gpu_index = -1,
                     VaeStats *stats = nullptr);
} // namespace ernie

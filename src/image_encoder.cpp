// SPDX-License-Identifier: MIT
#include "image_encoder.h"
#include "ernie_gelu.h"
#include <cmath>
#include <stdexcept>

namespace ernie
{
namespace
{
void check(int rc, const char *action)
{
    if (rc) throw std::runtime_error(std::string(action) + " failed: " + std::to_string(rc));
}

void validate_output(const ncnn::Mat &value, int width, int height, int channels, const char *name)
{
    if (value.empty() || value.dims != 3 || value.w != width || value.h != height ||
        value.c != channels || value.elemsize != 4u || value.elempack != 1)
        throw std::runtime_error(std::string("Unexpected VAE encoder ") + name + " shape or storage");
    for (int c = 0; c < value.c; ++c)
        for (int i = 0; i < value.w * value.h; ++i)
            if (!std::isfinite(value.channel(c)[i]))
                throw std::runtime_error(std::string("Non-finite VAE encoder ") + name);
}
}

VaeEncoding encode_vae(const ComponentFiles &component, const RgbImage &rgb,
                       const ncnn::Option &requested)
{
    if ((rgb.width != 32 && rgb.width != 64) || rgb.height != 32 ||
        rgb.pixels.size() != size_t(rgb.width) * size_t(rgb.height) * 3)
        throw std::invalid_argument("Reviewed VAE encoder shapes are only 32x32 and 64x32 RGB");
    if (component.param_text.empty() || component.param_text.find('\0') != std::string::npos ||
        component.weight_path.empty())
        throw std::invalid_argument("Invalid authenticated VAE encoder component");
    if (requested.use_vulkan_compute || requested.use_fp16_storage || requested.use_bf16_storage ||
        requested.use_fp16_arithmetic || requested.use_fp16_packed || requested.use_bf16_packed)
        throw std::invalid_argument("VAE encoder contract requires CPU FP32");

    ncnn::Mat input(rgb.width, rgb.height, 3);
    if (input.empty()) throw std::bad_alloc();
    // Preserve the frozen conversion order rather than replacing it with /127.5-1.
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < rgb.height; ++y)
            for (int x = 0; x < rgb.width; ++x)
            {
                const float value = float(rgb.pixels[(size_t(y) * rgb.width + x) * 3 + c]);
                input.channel(c)[y * rgb.width + x] = (value - 127.5f) * (1.f / 127.5f);
            }

    VaeEncoding result;
    ncnn::Net net;
    net.opt = requested;
    net.opt.use_packing_layout = false;
    net.opt.use_winograd_convolution = false;
    check(register_layers(net), "Register VAE encoder layers");
    check(net.load_param_mem(component.param_text.c_str()), "Load VAE encoder graph");
    check(net.load_model(component.weight_path.c_str()), "Load VAE encoder weights");
    auto ex = net.create_extractor();
    check(ex.input("in0", input), "Input VAE encoder RGB");
    check(ex.extract("out0", result.mean), "Extract VAE encoder mean");
    check(ex.extract("out1", result.packed), "Extract VAE encoder packed mean");
    check(ex.extract("out2", result.normalized), "Extract VAE encoder normalized latent");
    validate_output(result.mean, rgb.width / 8, rgb.height / 8, 32, "mean");
    validate_output(result.packed, rgb.width / 16, rgb.height / 16, 128, "packed");
    validate_output(result.normalized, rgb.width / 16, rgb.height / 16, 128, "normalized");
    return result;
}
}

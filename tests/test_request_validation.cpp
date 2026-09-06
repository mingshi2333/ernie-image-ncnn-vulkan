// SPDX-License-Identifier: MIT
#include <ernie/pipeline.h>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
ernie::GenerationRequest base()
{
    ernie::GenerationRequest request;
    request.model = "absent-model-request-validation";
    request.prompt = "cat";
    request.device = "cpu";
    request.precision = "fp32";
    return request;
}

void rejected(const std::string &expected, const std::function<void(ernie::GenerationRequest &)> &change)
{
    auto request = base();
    change(request);
    try
    {
        (void)ernie::generate(request);
    }
    catch (const std::invalid_argument &error)
    {
        if (std::string(error.what()).find(expected) != std::string::npos)
            return;
        throw std::runtime_error("Unexpected validation error: " + std::string(error.what()));
    }
    throw std::runtime_error("Invalid request reached model loading");
}
} // namespace

int main()
{
    try
    {
        rejected("Threads", [](auto &r) { r.threads = 0; });
        rejected("Threads", [](auto &r) { r.threads = 257; });
        rejected("GPU index", [](auto &r) { r.gpu_index = -2; });
        rejected("GPU index requires", [](auto &r) { r.gpu_index = 0; });
        rejected("CPU text", [](auto &r) { r.text_device = "vulkan"; });
        rejected("finite", [](auto &r) { r.strength = std::numeric_limits<float>::infinity(); });
        rejected("[0,1]", [](auto &r) { r.strength = -0.01f; });
        rejected("width*height*3", [](auto &r) { r.input_image = ernie::RgbImage{2, 2, {0, 1}}; });
        rejected("dimensions differ",
                 [](auto &r)
                 {
                     r.width = r.height = 16;
                     r.input_image = ernie::RgbImage{2, 2, std::vector<uint8_t>(12)};
                 });
        rejected("unsupported until the F2",
                 [](auto &r) { r.input_image = ernie::RgbImage{2, 2, std::vector<uint8_t>(12)}; });
        rejected("Vulkan VAE",
                 [](auto &r)
                 {
                     r.vae_device = "vulkan";
                     r.gpu_index = 0;
                 });
        std::cout
            << "Request validation rejects invalid and unavailable configurations before model loading\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

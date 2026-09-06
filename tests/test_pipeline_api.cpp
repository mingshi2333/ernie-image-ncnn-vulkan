// SPDX-License-Identifier: MIT
// An embedding application only includes the public API, not ncnn or PNG.
#include <ernie/pipeline.h>
#include <iostream>
#include <stdexcept>

int main()
{
    try
    {
        ernie::GenerationRequest request;
        if (request.threads != 4 || request.gpu_index != -1 || request.text_device != "cpu" ||
            request.input_image || request.strength != .5f || request.input_resize != "none" ||
            request.input_source_width || request.input_source_height ||
            request.input_alpha_background != std::array<uint8_t,3>{255,255,255} ||
            request.input_resize_background != std::array<uint8_t,3>{0,0,0})
            throw std::runtime_error("Public request defaults changed unexpectedly");
        ernie::GenerationRequest legacy{"model", "prompt", "cpu", "fp32", "cpu", "direct", 0, 0,
                                        8,       42,       "",    {},     "",    "",       ""};
        if (legacy.model != "model" || legacy.threads != 4 || legacy.input_image)
            throw std::runtime_error("Older positional request initialization is no longer compatible");
        const auto diagnostics = ernie::diagnose();
        if (!diagnostics.vulkan_compiled && !diagnostics.vulkan_devices.empty())
            throw std::runtime_error("A non-Vulkan build reported Vulkan devices");
        for (size_t i = 0; i < diagnostics.vulkan_devices.size(); ++i)
            if (diagnostics.vulkan_devices[i].index != int(i) || diagnostics.vulkan_devices[i].name.empty())
                throw std::runtime_error("Vulkan diagnostics returned an invalid device record");
        if (!diagnostics.vulkan_devices.empty() &&
            (diagnostics.default_gpu_index < 0 ||
             diagnostics.default_gpu_index >= int(diagnostics.vulkan_devices.size())))
            throw std::runtime_error("Vulkan diagnostics returned an invalid default device");
        request.model = "nonexistent-model-for-api-contract";
        request.prompt = "cat";
        request.width = 512;
        bool rejected = false, callback_called = false;
        try
        {
            ernie::generate(request, [&](const ernie::Progress &) { callback_called = true; });
        }
        catch (const std::invalid_argument &e)
        {
            rejected = std::string(e.what()).find("together") != std::string::npos;
        }
        if (!rejected || callback_called)
            throw std::runtime_error("Public API did not validate request before inference");
        request.height = 384;
        request.device = "cpu";
        request.precision = "bf16";
        rejected = false;
        try
        {
            ernie::generate(request);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Public API allowed unsupported CPU precision");
        std::cout << "Public C++ API builds without private headers and reports runtime capabilities\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

// SPDX-License-Identifier: MIT
#include <ernie/pipeline.h>
#include <iostream>
#include <stdexcept>
#if __has_include(<net.h>) || __has_include(<model_package.h>)
#error "Private implementation include paths leaked into the installed API"
#endif

int main()
{
    try
    {
        ernie::GenerationRequest request;
        request.device = "cpu";
        request.precision = "fp32";
        request.model = "missing-installed-test-model";
        request.prompt = "installation contract";
        request.width = 512; // Deliberately missing height: rejects before loading.
        request.input_image = ernie::RgbImage{1, 1, {255, 0, 0}};
        bool rejected = false, progress_called = false;
        try
        {
            ernie::generate(request, [&](const ernie::Progress&) { progress_called = true; });
        }
        catch (const std::invalid_argument& error)
        {
            rejected = std::string(error.what()).find("together") != std::string::npos;
        }
        if (!rejected || progress_called)
            throw std::runtime_error("Installed pipeline did not reject the request before inference");
        rejected = false;
        try { ernie::verify_model(request.model); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Installed tokenizer/package bridge did not reject a missing model");
        const auto info = ernie::diagnose();
        if (!info.vulkan_compiled && !info.vulkan_devices.empty())
            throw std::runtime_error("CPU installation reports Vulkan devices");
        std::cout << "Installed standard C++ API and package bridge passed; Vulkan compiled="
                  << info.vulkan_compiled << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

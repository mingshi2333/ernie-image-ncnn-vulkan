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
        std::cout << "Public C++ API builds without private headers and rejects invalid requests\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

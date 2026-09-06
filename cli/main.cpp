// SPDX-License-Identifier: MIT
#include "image_io.h"
#include "options.h"
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        const auto options = ernie::cli::parse_options(argc, argv);
        const auto &request = options.generation;
        if (options.help)
        {
            std::cout << ernie::cli::usage();
            return 0;
        }
        if (options.verify_only)
        {
            if (!request.model.empty())
                ernie::verify_model(request.model);
            if (!request.pe_model.empty())
                ernie::verify_pe_model(request.pe_model);
            std::cout << "Model verified\n";
            return 0;
        }
        if (!options.input.empty())
            throw std::invalid_argument("--input is recognized, but img2img generation is unsupported until "
                                        "the F2 runtime is integrated");
        const auto result = ernie::generate(
            request,
            [&](const ernie::Progress &p)
            {
                if (p.stage == "verify")
                    std::cout << "Model verified: " << p.seconds << " s";
                else if (p.stage == "text")
                    std::cout << "Text conditioned: " << p.current << " tokens, " << p.seconds
                              << " s, embeddings " << (request.embeddings.empty() ? "computed" : "loaded");
                else if (p.stage == "denoise")
                    std::cout << "Denoise " << p.current << '/' << p.total << ": " << p.seconds << " s";
                else
                    std::cout << p.stage << ' ' << p.current << '/' << p.total;
                std::cout << std::endl;
            });
        const auto write_start = std::chrono::steady_clock::now();
        ernie::cli::write_image(options.output, result.image);
        const auto write_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - write_start).count();
        if (!request.pe_model.empty())
            std::cout << "PE generated " << result.pe_generated_tokens << " tokens, "
                      << (result.pe_eos ? "EOS reached" : "token limit reached")
                      << "\nEnhanced prompt: " << result.prompt << '\n';
        std::cout << "VAE and image encode: " << result.vae_seconds + write_seconds << " s\n"
                  << "Saved " << result.image.width << 'x' << result.image.height
                  << " image: " << options.output.string() << '\n'
                  << "Total: " << result.elapsed_seconds + write_seconds << " s\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

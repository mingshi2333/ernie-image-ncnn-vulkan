// SPDX-License-Identifier: MIT
#include <ernie/pipeline.h>
#include "conditioning.h"
#include "model_package.h"
#include "options.h"
#include "tensor_io.h"
#include "text_encoder.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects_with(F action, const char* expected)
{
    try { action(); }
    catch (const std::exception& error)
    {
        if (std::string(error.what()).find(expected) != std::string::npos) return;
        throw std::runtime_error(std::string("Wrong file reached: ") + error.what());
    }
    throw std::runtime_error("Malformed fixture was accepted");
}
struct Temporary {
    fs::path path = fs::temp_directory_path() / fs::u8path(u8"ernie-\u6a21\u578b \u041f\u0443\u0442\u044c \U0001f5bc-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary() { require(fs::create_directory(path), "Cannot create native Unicode fixture"); }
    ~Temporary() { std::error_code ignored; fs::remove_all(path, ignored); }
};
}

int main()
{
    try
    {
        Temporary temporary;
        const auto& root = temporary.path;
        // Native filesystem paths create these fixtures independently of the
        // UTF-8 strings passed across the application and Rust boundaries.
        { std::ofstream(root / "model.cfg") << "text_layers 25\n"; }
        rejects_with([&] { ernie::verify_model(root.u8string()); }, "Missing model.cfg field");
        fs::remove(root / "model.cfg");
        { std::ofstream(root / "manifest.json") << "{\"schema_version\":3}"; }
        rejects_with([&] { ernie::verify_model(root.u8string()); }, "Unexpected schema-3 fields");
        rejects_with([&] { ernie::ModelPackage package(root); }, "Unexpected schema-3 fields");

        const std::string prompt = u8"\u7ea2\u8272\u82f9\u679c \u041f\u0440\u0438\u0432\u0435\u0442 \U0001f34e\n";
        const auto prompt_path = root / fs::u8path(u8"\u63d0\u793a\u8bcd \U0001f34e.txt");
        { std::ofstream(prompt_path, std::ios::binary) << prompt; }
        const auto output = root / fs::u8path(u8"\u56fe\u50cf \U0001f5bc.png");
        const auto report = root / fs::u8path(u8"\u62a5\u544a \U0001f5bc.json");
        std::vector<std::string> arguments{"ernie-image", "--model", root.u8string(),
            "--prompt-file", prompt_path.u8string(), "--output", output.u8string(), "--report-json", report.u8string()};
        std::vector<char*> pointers;
        for (auto& value : arguments) pointers.push_back(value.data());
        auto options = ernie::cli::parse_options(int(pointers.size()), pointers.data());
        require(options.generation.model == root.u8string() && options.generation.prompt == prompt &&
                options.output == output && options.report_json == report, "UTF-8 CLI path or prompt changed");

        ncnn::Mat frequencies(64); frequencies.fill(1.f);
        const auto frequency_path = root / fs::u8path(u8"\u9891\u7387 \U0001f5bc.f32");
        ernie::write_tensor(frequency_path, frequencies);
        auto read = ernie::read_tensor(frequency_path, 64);
        require(read[63] == 1.f, "Native tensor path changed");
        const auto text = ernie::text_constants(frequency_path.u8string(), 2);
        const auto image = ernie::dit_constants(frequency_path.u8string(), 1, 1, 1, 1);
        require(text.size() == 3 && text[0].h == 2 && image.size() == 3 && image[0].h == 2,
                "UTF-8 conditioning file changed");
        std::cout << "Native Unicode paths, UTF-8 API/Rust/CLI, prompt and tensor I/O passed; no model inference\n";
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}

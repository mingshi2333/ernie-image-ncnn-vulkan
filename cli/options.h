// SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include <ernie/pipeline.h>
#include <filesystem>
namespace ernie::cli
{
struct Options
{
    GenerationRequest generation;
    std::filesystem::path output, input, metrics_json, report_json;
    std::array<uint8_t, 3> background{255, 255, 255};
    std::string resize;
    bool background_explicit = false;
    bool help = false, verify_only = false, diagnose_only = false;
};
Options parse_options(int argc, char **argv);
const char *usage();
} // namespace ernie::cli

// SPDX-License-Identifier: MIT
#pragma once
#include <ernie/pipeline.h>
#include <filesystem>
namespace ernie::cli
{
struct Options
{
    GenerationRequest generation;
    std::filesystem::path output;
    bool help = false, verify_only = false;
};
Options parse_options(int argc, char **argv);
const char *usage();
} // namespace ernie::cli

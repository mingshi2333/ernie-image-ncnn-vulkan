// SPDX-License-Identifier: MIT
#pragma once
#include <ernie/pipeline.h>
#include <filesystem>
#include <string>
#include <vector>

namespace ernie::cli {
// Small completion record. Counters describe placement requests and charged
// cache weights, not complete process/allocator memory measurements.
std::string generation_report_json(const GenerationRequest&, const GenerationResult&,
    const std::vector<Progress>&, double image_write_seconds, bool allocation_instrumentation);
void write_generation_report(const std::filesystem::path&, const GenerationRequest&,
    const GenerationResult&, const std::vector<Progress>&, double image_write_seconds,
    bool allocation_instrumentation);
} // namespace ernie::cli

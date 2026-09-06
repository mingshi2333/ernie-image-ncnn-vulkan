// SPDX-License-Identifier: MIT
#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace ncnn { class Net; }
namespace ernie
{
// An already resolved graph and weight object. Package verification belongs to
// ModelPackage; component execution never parses a manifest or copies weights.
struct ComponentFiles
{
    std::string param_text;
    std::string weight_path;
    bool empty() const { return param_text.empty() || weight_path.empty(); }
};

// Legacy probe adapters. These bounded reads preserve existing directory CLI
// arguments; production generation obtains components from ModelPackage.
ComponentFiles component_files(const std::filesystem::path &directory, const std::string &stem);
std::vector<ComponentFiles> component_files(const std::vector<std::string> &directories,
                                            const std::string &stem);
void load_component(ncnn::Net &net, const ComponentFiles &files);
} // namespace ernie

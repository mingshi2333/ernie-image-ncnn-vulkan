// SPDX-License-Identifier: MIT
#include "component_files.h"
#include "net.h"
#include <fstream>
#include <stdexcept>
#include <utility>

namespace ernie
{
ComponentFiles component_files(const std::filesystem::path &directory, const std::string &stem)
{
    if (stem != "head" && stem != "block" && stem != "text")
        throw std::invalid_argument("Unknown component filename stem");
    const auto path = directory / (stem + ".ncnn.param");
    const auto size = std::filesystem::file_size(path);
    if (!size || size > 1024 * 1024)
        throw std::invalid_argument("Graph text must contain 1..1048576 bytes");
    std::string graph(size, '\0');
    std::ifstream input(path, std::ios::binary);
    if (!input.read(graph.data(), std::streamsize(size)) || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Cannot read the complete graph text");
    if (graph.find('\0') != std::string::npos)
        throw std::invalid_argument("Graph text contains NUL");
    return {std::move(graph), (directory / (stem + ".ncnn.bin")).string()};
}

std::vector<ComponentFiles> component_files(const std::vector<std::string> &directories,
                                            const std::string &stem)
{
    std::vector<ComponentFiles> result;
    result.reserve(directories.size());
    for (const auto &directory : directories)
        result.push_back(component_files(std::filesystem::path(directory), stem));
    return result;
}

void load_component(ncnn::Net &net, const ComponentFiles &files)
{
    if (files.empty() || files.param_text.size() > 1024 * 1024 ||
        files.param_text.find('\0') != std::string::npos ||
        files.weight_path.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid resolved graph or weight path");
    load_component_param(net, files);
    load_component_model(net, files);
}
void load_component_param(ncnn::Net &net, const ComponentFiles &files)
{
    if (files.param_text.empty() || files.param_text.size() > 1024 * 1024 || files.param_text.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid resolved graph");
    const int graph_status = net.load_param_mem(files.param_text.c_str());
    if (graph_status)
        throw std::runtime_error("Load component graph failed: " + std::to_string(graph_status));
}
void load_component_model(ncnn::Net &net, const ComponentFiles &files)
{
    if (files.weight_path.empty() || files.weight_path.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid resolved weight path");
    const int weight_status = net.load_model(files.weight_path.c_str());
    if (weight_status)
        throw std::runtime_error("Load component weights failed: " + std::to_string(weight_status));
}
} // namespace ernie

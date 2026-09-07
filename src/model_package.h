// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "model_config.h"
#include <filesystem>
#include <string>
namespace ernie {
class ModelPackage {
public:
    // WH=0 selects a sole instance. Schema3 instantiates the bounded runtime
    // shape contract from authenticated independently exported source templates.
    explicit ModelPackage(const std::filesystem::path &directory, int width=0,int height=0);
    ~ModelPackage();
    ModelPackage(const ModelPackage&)=delete;
    ModelPackage& operator=(const ModelPackage&)=delete;
    ModelPackage(ModelPackage&&) noexcept;
    ModelPackage& operator=(ModelPackage&&) noexcept;
    const ModelConfig &config() const { return config_; }
    const ModelConfig &source_config() const { return source_config_; }
    int schema() const { return schema_; }
    void select_text_tokens(std::size_t tokens);
    std::string file(const std::string &logical_name) const;
    bool has_file(const std::string &logical_name) const;
    ComponentFiles component(const std::string &logical_param,const std::string &kind) const;
private:
    void refresh_config();
    void *handle_=nullptr;
    ModelConfig config_{};
    ModelConfig source_config_{};
    int schema_=0;
};
}

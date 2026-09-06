// SPDX-License-Identifier: MIT
#pragma once
#include "component_files.h"
#include "model_config.h"
#include <filesystem>
#include <string>
namespace ernie {
class ModelPackage {
public:
    // WH=0 selects a sole instance. Multi-instance schema3 requires explicit WH.
    explicit ModelPackage(const std::filesystem::path &directory, int width=0,int height=0);
    ~ModelPackage();
    ModelPackage(const ModelPackage&)=delete;
    ModelPackage& operator=(const ModelPackage&)=delete;
    ModelPackage(ModelPackage&&) noexcept;
    ModelPackage& operator=(ModelPackage&&) noexcept;
    const ModelConfig &config() const { return config_; }
    int schema() const { return schema_; }
    std::string file(const std::string &logical_name) const;
    ComponentFiles component(const std::string &logical_param,const std::string &kind) const;
private:
    void *handle_=nullptr;
    ModelConfig config_{};
    int schema_=0;
};
}

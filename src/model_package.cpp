// SPDX-License-Identifier: MIT
#include "model_package.h"
#include "shape_graph.h"
#include <array>
#include <fstream>
#include <stdexcept>
#include <utility>
extern "C" {
void *ernie_model_package_open(const unsigned char *,size_t,int,int,unsigned char *,size_t);
void ernie_model_package_destroy(void *);
int ernie_model_package_config(const void *,int *);
int ernie_model_package_source_config(const void *,int *);
int ernie_model_package_select_text_tokens(void *,size_t,unsigned char *,size_t);
int ernie_model_package_file(const void *,const unsigned char *,size_t,unsigned char *,size_t);
int ernie_model_package_has_file(const void *,const unsigned char *,size_t);
}
namespace ernie {
ModelPackage::ModelPackage(const std::filesystem::path &directory,int w,int h){
    auto root=std::filesystem::absolute(directory).string();std::array<unsigned char,4096> error{};
    handle_=ernie_model_package_open(reinterpret_cast<const unsigned char*>(root.data()),root.size(),w,h,error.data(),error.size());
    if(!handle_)throw std::runtime_error(std::string("Cannot open model package: ") + reinterpret_cast<const char*>(error.data()));
    try { refresh_config(); }
    catch(...){ernie_model_package_destroy(handle_);handle_=nullptr;throw;}
}
void ModelPackage::refresh_config(){
    int values[4]{};schema_=ernie_model_package_config(handle_,values);
    config_={values[0],values[1],values[2],values[3]};
    if(schema_<1||schema_>3 || ernie_model_package_source_config(handle_,values)!=schema_)
        throw std::runtime_error("Invalid package schema");
    source_config_={values[0],values[1],values[2],values[3]};
    if(schema_==3)validate_runtime_model_config(config_);else validate_model_config(config_);
    validate_model_config(source_config_);
}
void ModelPackage::select_text_tokens(std::size_t tokens){
    std::array<unsigned char,4096> error{};
    if(ernie_model_package_select_text_tokens(handle_,tokens,error.data(),error.size()))
        throw std::invalid_argument(reinterpret_cast<const char*>(error.data()));
    refresh_config();
}
ModelPackage::~ModelPackage(){ernie_model_package_destroy(handle_);}
ModelPackage::ModelPackage(ModelPackage&& other) noexcept:handle_(std::exchange(other.handle_,nullptr)),config_(other.config_),source_config_(other.source_config_),schema_(other.schema_){}
ModelPackage& ModelPackage::operator=(ModelPackage&& other) noexcept{if(this!=&other){ernie_model_package_destroy(handle_);handle_=std::exchange(other.handle_,nullptr);config_=other.config_;source_config_=other.source_config_;schema_=other.schema_;}return *this;}
std::string ModelPackage::file(const std::string &name)const{
    std::array<unsigned char,16384> output{};
    if(ernie_model_package_file(handle_,reinterpret_cast<const unsigned char*>(name.data()),name.size(),output.data(),output.size()))throw std::invalid_argument("Unknown package file or invalid package handle");
    return reinterpret_cast<const char*>(output.data());
}
bool ModelPackage::has_file(const std::string &name)const{
    return ernie_model_package_has_file(handle_,reinterpret_cast<const unsigned char*>(name.data()),name.size())==1;
}
ComponentFiles ModelPackage::component(const std::string &name,const std::string &kind)const{
    constexpr const char *suffix=".param";
    if(name.size()<6||name.substr(name.size()-6)!=suffix)throw std::invalid_argument("Expected logical param filename");
    auto path=file(name);auto size=std::filesystem::file_size(path);
    if(size>1024*1024)throw std::invalid_argument("Graph exceeds text limit");
    std::string text(size,'\0');std::ifstream input(path,std::ios::binary);
    if(!input.read(text.data(),size))throw std::runtime_error("Cannot read graph object");
    if(text.find('\0')!=std::string::npos)throw std::invalid_argument("Embedded NUL in graph");
    // Existing legacy packages preserve their former static graph loader behavior.
    // Schema3 requires full topology hash validation even for same-shape instantiation.
    if(schema_==3 && kind!="vae-encoder")text=instantiate_shape_graph(kind,text,source_config_,config_);
    if(kind=="vae-encoder" && name!="vae/encoder.ncnn.param")throw std::invalid_argument("Unexpected VAE encoder logical name");
    return {std::move(text),file(name.substr(0,name.size()-6)+".bin")};
}
}

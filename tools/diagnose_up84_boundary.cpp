// SPDX-License-Identifier: MIT
// Diagnostic only: retain one boundary, then verify unchanged final output externally.
#include "ernie_gelu.h"
#include "gpu.h"
#include "pipelinecache.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
namespace fs=std::filesystem;
static void check(int code){if(code)throw std::runtime_error("ncnn operation failed "+std::to_string(code));}
static ncnn::Mat read(const fs::path& p,int w,int h){
 if(fs::file_size(p)!=size_t(w)*h*4)throw std::runtime_error("Bad input bytes");
 ncnn::Mat x(w,h);std::ifstream f(p,std::ios::binary);if(!f.read((char*)x.data,size_t(w)*h*4))throw std::runtime_error("Read failed");return x;
}
static void save(const fs::path& p,const ncnn::Mat& x){
 if(fs::exists(p)||x.empty()||x.elempack!=1||x.elemsize!=4)throw std::runtime_error("Bad/existing output");
 std::ofstream f(p,std::ios::binary);for(int c=0;c<x.c;++c)f.write((const char*)(const float*)x.channel(c),size_t(x.w)*x.h*x.d*4);
 if(!f)throw std::runtime_error("Write failed");
}
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"MODEL FIXTURE NEW_OUTPUT\n";return 2;}
 int status=0;
 try{
  fs::path model=argv[1],fixture=argv[2],output=argv[3];
  if(fs::exists(output))throw std::runtime_error("Output exists");
  fs::create_directories(output);
  std::vector<ncnn::Mat> host{read(fixture/"in0.f32",4096,4160)};
  for(int i=1;i<=6;++i)host.push_back(read(fixture/("in"+std::to_string(i)+".f32"),4096,1));
  host.push_back(read(fixture/"in7.f32",128,4160));host.push_back(read(fixture/"in8.f32",128,4160));host.push_back(read(fixture/"in9.f32",4160,4160));
  ncnn::create_gpu_instance();if(ncnn::get_gpu_count()<1)throw std::runtime_error("No device");
  auto device=ncnn::get_gpu_device(ncnn::get_default_gpu_index());
  ncnn::PipelineCache cache(device);ncnn::VkBlobAllocator blob(device);ncnn::VkStagingAllocator staging(device);
  ncnn::Option option;option.num_threads=4;option.use_vulkan_compute=true;
  option.use_fp16_storage=option.use_fp16_packed=option.use_fp16_arithmetic=false;
  option.use_bf16_storage=option.use_bf16_packed=false;option.use_weights_in_host_memory=false;
  option.pipeline_cache=&cache;option.blob_vkallocator=option.workspace_vkallocator=&blob;option.staging_vkallocator=&staging;
  std::vector<ncnn::VkMat> inputs(host.size());
  {ncnn::VkCompute cmd(device);for(size_t i=0;i<host.size();++i)cmd.record_upload(host[i],inputs[i],option);check(cmd.submit_and_wait());}
  ncnn::Net net;net.opt=option;net.set_vulkan_device(device);check(ernie::register_layers(net));
  check(net.load_param((model/"block.ncnn.param").c_str()));check(net.load_model((model/"block.ncnn.bin").c_str()));
  for(auto layer:net.layers())if(!layer->support_vulkan&&layer->type!="Input"&&layer->type!="Split")throw std::runtime_error("NonVulkan layer");
  auto extractor=net.create_extractor();extractor.set_blob_vkallocator(&blob);extractor.set_workspace_vkallocator(&blob);extractor.set_staging_vkallocator(&staging);
  for(size_t i=0;i<inputs.size();++i)check(extractor.input(("in"+std::to_string(i)).c_str(),inputs[i]));
  ncnn::VkMat input81,boundary,final;
  {ncnn::VkCompute cmd(device);check(extractor.extract("81",input81,cmd));check(extractor.extract("84",boundary,cmd));check(extractor.extract("out0",final,cmd));check(cmd.submit_and_wait());}
  ncnn::Mat hi,hb,hf;ncnn::Option plain=option;plain.use_packing_layout=false;
  {ncnn::VkCompute cmd(device);cmd.record_download(input81,hi,plain);cmd.record_download(boundary,hb,plain);cmd.record_download(final,hf,plain);check(cmd.submit_and_wait());}
  save(output/"81.f32",hi);save(output/"84.f32",hb);save(output/"out0.f32",hf);
  if(hi.w!=4096||hi.h!=4160||hi.d!=1||hi.c!=1||hb.w!=12288||hb.h!=4160||hb.d!=1||hb.c!=1||hf.w!=4096||hf.h!=4160||hf.d!=1||hf.c!=1)throw std::runtime_error("Wrong full layout");
  std::cout<<"UP84_COMPLETE_LAYOUT_OK\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 ncnn::destroy_gpu_instance();return status;
}

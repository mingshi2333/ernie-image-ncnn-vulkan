// SPDX-License-Identifier: MIT
// Standalone opt-in fixed-shape arithmetic probe. Does not register a production layer.
#include "gpu.h"
#include "pipeline.h"
#include "command.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
namespace fs=std::filesystem;
static void check(int c){if(c)throw std::runtime_error("ncnn operation failed "+std::to_string(c));}
static std::string source(const fs::path&p){std::ifstream f(p);if(!f)throw std::runtime_error("shader missing");return {std::istreambuf_iterator<char>(f),{}};}
static ncnn::Mat read(const fs::path&p,int w,int h){if(fs::file_size(p)!=size_t(w)*h*4)throw std::runtime_error("bad tensor size");ncnn::Mat x(w,h);std::ifstream f(p,std::ios::binary);if(!f.read((char*)x.data,size_t(w)*h*4))throw std::runtime_error("read failed");for(int i=0;i<w*h;++i)if(!std::isfinite(((float*)x.data)[i]))throw std::runtime_error("nonfinite");return x;}
int main(int argc,char**argv){
 if(argc!=6){std::cerr<<"X WEIGHT_KN M SHADER_DIR NEW_OUTPUT\n";return 2;}
 int rc=0;
 try{
  size_t used=0;int m=std::stoi(argv[3],&used);if(used!=std::string(argv[3]).size()||m<1||m>10)throw std::runtime_error("M must be 1..10");
  const int n=4096,k=12288,s=32;fs::path output=argv[5];if(fs::exists(output))throw std::runtime_error("output exists");
  auto hx=read(argv[1],k*m,1),hw=read(argv[2],n*k,1);
  ncnn::create_gpu_instance();if(ncnn::get_gpu_count()<1)throw std::runtime_error("no device");auto dev=ncnn::get_gpu_device(ncnn::get_default_gpu_index());
  ncnn::VkBlobAllocator blob(dev);ncnn::VkStagingAllocator staging(dev);ncnn::Option opt;opt.num_threads=2;opt.use_vulkan_compute=true;opt.use_packing_layout=false;opt.use_fp16_storage=opt.use_fp16_packed=opt.use_fp16_arithmetic=opt.use_bf16_storage=opt.use_bf16_packed=false;opt.blob_vkallocator=opt.workspace_vkallocator=&blob;opt.staging_vkallocator=&staging;
  ncnn::Pipeline partial(dev),merge(dev);partial.set_local_size_xyz(64,1,1);merge.set_local_size_xyz(64,1,1);
  std::vector<uint32_t> spirv;auto a=source(fs::path(argv[4])/"down_splitk.comp");check(ncnn::compile_spirv_module(a.c_str(),opt,spirv));check(partial.create(spirv.data(),spirv.size()*4,{}));spirv.clear();auto b=source(fs::path(argv[4])/"down_merge.comp");check(ncnn::compile_spirv_module(b.c_str(),opt,spirv));check(merge.create(spirv.data(),spirv.size()*4,{}));
  ncnn::VkMat x,w,temp(n*m*s*2,size_t(4),1,&blob),y(n,m,size_t(4),1,&blob);if(temp.empty()||y.empty())throw std::runtime_error("allocation failed");
  {ncnn::VkCompute cmd(dev);cmd.record_upload(hx,x,opt);cmd.record_upload(hw,w,opt);check(cmd.submit_and_wait());}
  if(x.elempack!=1||w.elempack!=1||x.h!=1||w.h!=1)throw std::runtime_error("raw shader requires flat scalar uploads");
  std::vector<ncnn::vk_constant_type> constants(4);constants[0].u32=m;constants[1].u32=n;constants[2].u32=k;constants[3].u32=s;
  ncnn::Mat result;
  {ncnn::VkCompute cmd(dev);ncnn::VkMat grid;grid.w=n;grid.h=m;grid.c=s;cmd.record_pipeline(&partial,{x,w,temp},constants,grid);grid.c=1;cmd.record_pipeline(&merge,{temp,y},constants,grid);cmd.record_download(y,result,opt);check(cmd.submit_and_wait());}
  if(result.w!=n||result.h!=m||result.elemsize!=4||result.elempack!=1)throw std::runtime_error("bad output layout");std::ofstream f(output,std::ios::binary);if(!f.write((char*)result.data,size_t(n)*m*4))throw std::runtime_error("write failed");
  std::cout<<"{\"m\":"<<m<<",\"n\":4096,\"k\":12288,\"splits\":32,\"scope\":\"diagnostic_only\"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';rc=1;}
 ncnn::destroy_gpu_instance();return rc;
}

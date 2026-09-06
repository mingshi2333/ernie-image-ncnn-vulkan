// SPDX-License-Identifier: MIT
// Diagnostic only. Fixed block15 down projection; no production registration.
#pragma once
#include "gemm.h"
#include "pipeline.h"
#include "command.h"
#include "gpu.h"
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstring>
static std::string q2_shader_directory;
static std::string q2_screen_weight;
class Q2SplitDown final : public ncnn::Gemm {
public:
 Q2SplitDown(){support_vulkan=true;support_vulkan_packing=true;}
 int load_param(const ncnn::ParamDict& p) override {
  int r=Gemm::load_param(p);if(r)return r;
  if(alpha!=1||beta!=1||transA!=0||transB!=1||constantA!=0||constantB!=1||constantC!=1||constantM!=4160||constantN!=4096||constantK!=12288||constant_broadcast_type_C!=-1||output_N1M||output_elempack||output_elemtype||output_transpose||quantize_term)return -1;
  return 0;
 }
 int create_pipeline(const ncnn::Option& opt) override {
  if(!opt.use_vulkan_compute||opt.use_fp16_storage||opt.use_fp16_packed||opt.use_fp16_arithmetic||opt.use_bf16_storage||opt.use_bf16_packed)return -1;
  partial.reset(new ncnn::Pipeline(vkdev));merge.reset(new ncnn::Pipeline(vkdev));
  partial->set_local_size_xyz(64,1,1);merge->set_local_size_xyz(64,1,1);
  for(auto item: {std::make_pair(partial.get(),"down_splitk_rows.comp"),std::make_pair(merge.get(),"down_merge_rows.comp")}){
   std::ifstream f(q2_shader_directory+"/"+item.second);if(!f)return -1;
   std::string code{std::istreambuf_iterator<char>(f),{}};std::vector<uint32_t> spirv;
   int r=ncnn::compile_spirv_module(code.c_str(),opt,spirv);if(r)return r;
   r=item.first->create(spirv.data(),spirv.size()*4,{});if(r)return r;
  }return 0;
 }
 int destroy_pipeline(const ncnn::Option&) override {partial.reset();merge.reset();return 0;}
 int upload_model(ncnn::VkTransfer& cmd,const ncnn::Option& opt) override {
  if(B_data.w!=12288||B_data.h!=4096||B_data.elempack!=1||B_data.elemsize!=4||!C_data.empty())return -1;
  ncnn::Mat flat(12288*4096,1);if(flat.empty())return -100;
  const float* b=(const float*)B_data.data;float* w=(float*)flat.data;
  for(int k=0;k<12288;++k)for(int n=0;n<4096;++n)w[size_t(k)*4096+n]=b[size_t(n)*12288+k];
  // Independently authenticated complete KN screen weight, not a guessed stream offset.
  std::ifstream reference(q2_screen_weight,std::ios::binary);if(!reference)return -1;
  std::vector<char> bytes(65536);const char* actual=(const char*)flat.data;
  for(size_t offset=0;offset<size_t(12288)*4096*4;offset+=bytes.size()){
   if(!reference.read(bytes.data(),bytes.size())||std::memcmp(bytes.data(),actual+offset,bytes.size()))return -1;
  }
  if(reference.peek()!=std::char_traits<char>::eof())return -1;
  std::cerr<<"Q2SplitDown complete_KN_weight_bitwise_equal=50331648\n";
  cmd.record_upload(flat,weight,opt);
  return weight.elempack==1&&weight.elemsize==4&&weight.h==1 ? 0 : -1;
 }
 int forward(const ncnn::VkMat& input,ncnn::VkMat& output,ncnn::VkCompute& cmd,const ncnn::Option& opt) const override {
  if(input.dims!=2||input.w!=12288||input.h*input.elempack!=4160||input.elembits()!=32)return -1;
  ncnn::VkMat scalar;vkdev->convert_packing(input,scalar,1,1,cmd,opt);
  if(scalar.w!=12288||scalar.h!=4160||scalar.elempack!=1||scalar.elemsize!=4)return -1;
  output.create(4096,4160,size_t(4),1,opt.blob_vkallocator);
  ncnn::VkMat temp(4096*16*32*2,size_t(4),1,opt.workspace_vkallocator);
  if(output.empty()||temp.empty())return -100;
  std::vector<unsigned char> coverage(4160,0);int submissions=0;
  for(int start=0;start<4160;start+=16){
   const int count=std::min(16,4160-start);
   for(int row=start;row<start+count;++row)if(++coverage[row]!=1)return -1;
   std::vector<ncnn::vk_constant_type> constants(5);
   constants[0].u32=count;constants[1].u32=4096;constants[2].u32=12288;constants[3].u32=32;constants[4].u32=start;
   ncnn::VkMat grid;grid.w=4096;grid.h=count;grid.c=32;
   cmd.record_pipeline(partial.get(),{scalar,weight,temp},constants,grid);
   grid.c=1;cmd.record_pipeline(merge.get(),{temp,output},constants,grid);
   // Finish all reads before reusing the bounded temporary, retaining all VkMat owners.
   int r=cmd.submit_and_wait();if(r)return r;r=cmd.reset();if(r)return r;++submissions;
  }
  if(std::any_of(coverage.begin(),coverage.end(),[](unsigned char x){return x!=1;}))return -1;
  std::cerr<<"Q2SplitDown rows=4160 covered_once=4160 chunks="<<submissions<<" chunk_rows=16 temp_bytes="<<4096*16*32*2*4<<"\n";
  return 0;
 }
private:
 std::unique_ptr<ncnn::Pipeline> partial,merge;
 ncnn::VkMat weight;
};
static ncnn::Layer* q2_create(void*){return new Q2SplitDown;}

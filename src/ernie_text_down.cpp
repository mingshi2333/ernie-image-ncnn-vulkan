// SPDX-License-Identifier: MIT
#include "ernie_text_down.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ernie
{
namespace
{
// Complete reviewed graphs: bucket64 SHA a9776b29..., bucket2048 SHA ba3ca63e....
// BUCKET occurs only at the fields that differ between those two original exports.
constexpr const char* reviewed = R"graph(7767517
58 75
Input in0 0 1 in0
Split splitncnn_0 1 2 in0 1 2
Input in1 0 1 in1
Input in2 0 1 in2
Input in3 0 1 in3
RMSNorm rmsn_8 1 1 2 6 0=3072 1=1.000000e-5 2=1
Split splitncnn_1 1 3 6 7 8 9
Gemm gemm_0 1 1 9 10 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=4096 9=3072
Reshape reshape_10 1 1 10 11 0=128 1=32 2=BUCKET
Permute transpose_14 1 1 11 12 0=2
Gemm gemm_1 1 1 8 13 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=1024 9=3072
Reshape reshape_11 1 1 13 14 0=128 1=8 2=BUCKET
Permute transpose_15 1 1 14 15 0=2
Gemm gemm_2 1 1 7 16 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=1024 9=3072
Reshape reshape_12 1 1 16 17 0=128 1=8 2=BUCKET
Permute transpose_16 1 1 17 18 0=2
Slice tensor_split_0 1 2 12 19 20 -23300=2,64,-233 1=2
Split splitncnn_3 1 2 20 21 22
Split splitncnn_2 1 2 19 23 24
Reshape unsqueeze_18 1 1 in1 25 0=128 1=BUCKET 2=1
Reshape unsqueeze_19 1 1 in2 26 0=128 1=BUCKET 2=1
Slice tensor_split_1 1 2 25 27 28 -23300=2,64,-233 1=2
Split splitncnn_5 1 2 28 29 30
Split splitncnn_4 1 2 27 31 32
Slice tensor_split_2 1 2 26 33 34 -23300=2,64,-233 1=2
Split splitncnn_7 1 2 34 35 36
Split splitncnn_6 1 2 33 37 38
BinaryOp mul_0 2 1 21 37 39 0=2
BinaryOp mul_1 2 1 23 31 40 0=2
BinaryOp sub_2 2 1 40 39 41 0=1
BinaryOp mul_3 2 1 24 35 42 0=2
BinaryOp mul_4 2 1 22 29 43 0=2
BinaryOp add_5 2 1 43 42 44 0=0
Concat cat_0 2 1 41 44 45 0=2
Slice tensor_split_3 1 2 15 46 47 -23300=2,64,-233 1=2
Split splitncnn_9 1 2 47 48 49
Split splitncnn_8 1 2 46 50 51
BinaryOp mul_6 2 1 48 38 52 0=2
BinaryOp mul_7 2 1 50 32 53 0=2
BinaryOp sub_8 2 1 53 52 54 0=1
BinaryOp mul_9 2 1 51 36 55 0=2
BinaryOp mul_10 2 1 49 30 56 0=2
BinaryOp add_11 2 1 56 55 57 0=0
Concat cat_1 2 1 54 57 58 0=2
SDPA sdpa_20 4 1 45 58 18 in3 59 5=1
Permute transpose_17 1 1 59 60 0=2
Reshape reshape_13 1 1 60 61 0=4096 1=BUCKET
Gemm gemm_3 1 1 61 62 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=3072 9=4096
BinaryOp add_12 2 1 1 62 63 0=0
Split splitncnn_10 1 2 63 64 65
RMSNorm rmsn_9 1 1 65 66 0=3072 1=1.000000e-5 2=1
Split splitncnn_11 1 2 66 67 68
Gemm gemm_4 1 1 68 69 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=9216 9=3072
Swish silu_7 1 1 69 70
Gemm gemm_5 1 1 67 71 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=9216 9=3072
BinaryOp mul_13 2 1 70 71 72 0=2
Gemm gemm_6 1 1 72 73 10=-1 2=0 3=1 4=0 5=1 6=1 7=BUCKET 8=3072 9=9216
BinaryOp add_14 2 1 64 73 out0 0=0
)graph";
std::vector<std::string> tokens(const std::string& text)
{
    std::istringstream input(text); std::vector<std::string> result; std::string word;
    while (input >> word) result.push_back(word);
    return result;
}
bool fp32_cpu(const ncnn::Option& opt)
{
    return !opt.use_vulkan_compute && !opt.use_fp16_storage && !opt.use_fp16_packed &&
           !opt.use_fp16_arithmetic && !opt.use_bf16_storage && !opt.use_bf16_packed;
}
class ErnieTextDown final : public ncnn::Layer
{
    std::unique_ptr<ncnn::Layer> inner{ncnn::create_layer("InnerProduct")};
public:
    ErnieTextDown() { one_blob_only = true; support_packing = false; }
    int load_param(const ncnn::ParamDict& pd) override
    {
        for (int i=0;i<3;++i) if (pd.type(i)!=2) return -1;
        for (int i=3;i<NCNN_MAX_PARAM_COUNT;++i) if (pd.type(i)) return -1;
        if (!inner || pd.get(0,0)!=3072 || pd.get(1,-1)!=0 || pd.get(2,0)!=28311552)
            return -1;
        return inner->load_param(pd);
    }
    int load_model(const ncnn::ModelBin& mb) override
    {
        // Gemm transB=1 reads [N,K]; InnerProduct reads the same N*K tagged values.
        ncnn::Mat weight=mb.load(28311552,0);
        if (weight.dims!=1 || weight.w!=28311552 || weight.elempack!=1 || weight.elemsize!=4u)
            return -1;
        for (int i=0;i<weight.w;++i) if (!std::isfinite(weight[i])) return -1;
        ncnn::ModelBinFromMatArray checked(&weight);
        return inner->load_model(checked);
    }
    int create_pipeline(const ncnn::Option& opt) override
    { return fp32_cpu(opt) ? inner->create_pipeline(opt) : -1; }
    int destroy_pipeline(const ncnn::Option& opt) override { return inner->destroy_pipeline(opt); }
    int forward(const ncnn::Mat& bottom,ncnn::Mat& top,const ncnn::Option& opt) const override
    {
        if (!fp32_cpu(opt) || bottom.dims!=2 || bottom.w!=9216 || bottom.h<1 || bottom.h>2048 ||
            bottom.elempack!=1 || bottom.elemsize!=4u) return -1;
        top.create(3072,bottom.h,4u,opt.blob_allocator);
        if (top.empty()) return -100;
        for (int r=0;r<bottom.h;++r)
        {
            const float* data=bottom.row(r);
            for (int i=0;i<9216;++i) if (!std::isfinite(data[i])) return -1;
            ncnn::Mat row(9216,const_cast<float*>(data)),result,plain;
            if (inner->forward(row,result,opt)) return -1;
            ncnn::convert_packing(result,plain,1,opt);
            if (plain.dims!=1 || plain.w!=3072 || plain.elemsize!=4u) return -1;
            for (int i=0;i<3072;++i) if (!std::isfinite(plain[i])) return -1;
            std::memcpy(top.row(r),plain.data,3072*sizeof(float));
        }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(ErnieTextDown)
}
std::string vector_text_down_graph(const std::string& graph,int bucket)
{
    if (graph.size()>65536 || (bucket!=64 && bucket!=2048))
        throw std::invalid_argument("Unreviewed vector text graph or bucket");
    std::string expected=reviewed;std::size_t pos=0;
    while ((pos=expected.find("BUCKET",pos))!=std::string::npos)
        expected.replace(pos,6,std::to_string(bucket));
    if (tokens(graph)!=tokens(expected)) throw std::invalid_argument("Complete text graph differs from reviewed template");
    std::istringstream input(expected);std::string line,result;int changed=0;
    while (std::getline(input,line))
    {
        if (line.rfind("Gemm gemm_6 ",0)==0)
        { line="ErnieTextDown gemm_6 1 1 72 73 0=3072 1=0 2=28311552"; ++changed; }
        result+=line+'\n';
    }
    if (changed!=1) throw std::logic_error("Missing reviewed down projection");
    return result;
}
void validate_text_down_weights(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open text weights");
    const std::uint64_t size=std::filesystem::file_size(path);std::uint64_t offset=0;
    auto skip=[&](std::uint64_t bytes) {
        if (offset>size || bytes>size-offset) throw std::invalid_argument("Truncated text weight stream");
        offset+=bytes;file.seekg(static_cast<std::streamoff>(offset));
        if (!file) throw std::runtime_error("Cannot seek text weight stream");
    };
    auto matrix=[&](std::uint64_t count) {
        if (offset+4>size) throw std::invalid_argument("Missing text weight tag");
        unsigned char raw[4];
        if (!file.read(reinterpret_cast<char*>(raw),4)) throw std::runtime_error("Cannot read text weight tag");
        const std::uint32_t tag=raw[0] | std::uint32_t(raw[1])<<8 | std::uint32_t(raw[2])<<16 | std::uint32_t(raw[3])<<24;
        offset+=4;
        if (tag!=0 && tag!=0x01348b83u) throw std::invalid_argument("Unsupported text weight storage tag");
        skip(count*(tag==0?4:2));
    };
    skip(3072*4); // input RMSNorm, raw FP32 affine weights
    matrix(4096ull*3072);matrix(1024ull*3072);matrix(1024ull*3072);matrix(3072ull*4096);
    skip(3072*4); // post-attention RMSNorm
    matrix(9216ull*3072);matrix(9216ull*3072);matrix(3072ull*9216);
    if (offset!=size) throw std::invalid_argument("Unexpected trailing text weights");
}
int register_text_down(ncnn::Net& net)
{ return net.register_custom_layer("ErnieTextDown",ErnieTextDown_layer_creator); }
}

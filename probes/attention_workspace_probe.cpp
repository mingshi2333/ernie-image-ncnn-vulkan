// SPDX-License-Identifier: MIT
#include "ernie_attention.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#if NCNN_VULKAN
class LimitedWorkspace : public ncnn::VkBlobAllocator
{
public:
    using ncnn::VkBlobAllocator::fastMalloc;
    explicit LimitedWorkspace(const ncnn::VulkanDevice* device) : VkBlobAllocator(device) {}
    ncnn::VkBufferMemory* fastMalloc(size_t size) override
    {
        largest = std::max(largest, size);
        if (size > 300*1024) return nullptr;
        return ncnn::VkBlobAllocator::fastMalloc(size);
    }
    size_t largest = 0;
};
#endif

#if NCNN_VULKAN
bool run_case(const char* name, int keys, int groups, int output_width, int mask_channels, int query_rows)
{
    constexpr int width=128, queries=257, heads=2;
    ncnn::Mat q(width,queries,heads), k(width,keys,groups), v(output_width,keys,groups), mask;
    if (mask_channels == 1) mask.create(keys,queries);
    if (mask_channels > 1) mask.create(keys,queries,mask_channels);
    for (int h=0; h<heads; ++h)
    {
        for (int i=0; i<queries; ++i)
            for (int d=0; d<width; ++d) q.channel(h).row(i)[d] = float((i*17+d*7+h*11)%53-26)/64.f;
    }
    for (int h=0; h<groups; ++h)
        for (int j=0; j<keys; ++j)
        {
            for (int d=0; d<width; ++d)
                k.channel(h).row(j)[d] = float((j*13+d*19+h*5)%67-33)/64.f;
            for (int d=0; d<output_width; ++d)
                v.channel(h).row(j)[d] = d==0 ? 1.f : float((j*31+d*3+h*7)%131-65)/128.f;
        }
    for (int c=0; c<mask_channels; ++c)
        for (int i=0; i<queries; ++i)
            for (int j=0; j<keys; ++j)
            {
                float* row=mask_channels==1 ? mask.row(i) : mask.channel(c).row(i);
                row[j]=j<keys-(i+c*7)%31 ? 0.f : -1e30f;
            }
    ncnn::Mat expected(output_width,queries,heads);
    for (int h=0; h<heads; ++h)
        for (int i=0; i<queries; ++i)
        {
            const int visible=mask_channels==0 ? keys : keys-(i+(mask_channels>1?h:0)*7)%31;
            const int group=h/(heads/groups);
            std::vector<double> probabilities(visible);
            double maximum=-1e100, total=0.;
            for (int j=0; j<visible; ++j)
            {
                double dot=0.;
                for (int d=0; d<width; ++d) dot+=double(q.channel(h).row(i)[d])*k.channel(group).row(j)[d];
                probabilities[j]=dot/std::sqrt(double(width)); maximum=std::max(maximum,probabilities[j]);
            }
            for (double& p:probabilities) { p=std::exp(p-maximum); total+=p; }
            for (int d=0; d<output_width; ++d)
            {
                double sum=0.;
                for (int j=0; j<visible; ++j) sum+=probabilities[j]*v.channel(group).row(j)[d];
                expected.channel(h).row(i)[d]=float(sum/total);
            }
        }
    LimitedWorkspace workspace(ncnn::get_gpu_device());
    ncnn::Mat actual, unsliced;
    uint64_t submissions=0;
    for (bool bounded : {true,false})
    {
        ncnn::Net net;
        net.opt.use_vulkan_compute=true;
        net.opt.use_fp16_storage=net.opt.use_fp16_packed=net.opt.use_fp16_arithmetic=false;
        net.opt.use_bf16_storage=net.opt.use_bf16_packed=false;
        net.opt.num_threads=4;
        if (bounded) net.opt.workspace_vkallocator=&workspace;
        if (ernie::register_attention(net,bounded)) throw std::runtime_error("Register attention");
        const char* graph=mask_channels ?
            "7767517\n5 5\nInput q 0 1 q\nInput k 0 1 k\nInput v 0 1 v\nInput mask 0 1 mask\nSDPA attention 4 1 q k v mask out 5=1\n" :
            "7767517\n4 4\nInput q 0 1 q\nInput k 0 1 k\nInput v 0 1 v\nSDPA attention 3 1 q k v out 5=0\n";
        alignas(4) const unsigned char weights[4]={};
        if (net.load_param_mem(graph)||net.load_model(weights)) throw std::runtime_error("Load attention");
        ernie::set_attention_query_rows(net, query_rows);
        auto ex=net.create_extractor();
        if (bounded) ex.set_workspace_vkallocator(&workspace);
        if (ex.input("q",q)||ex.input("k",k)||ex.input("v",v)||(mask_channels&&ex.input("mask",mask))) throw std::runtime_error("Input attention");
        const int rc=ex.extract("out",bounded?actual:unsliced);
        if (rc)
        {
            std::cout << "{\"return_code\":" << rc << ",\"largest_workspace_request\":" << workspace.largest << ",\"passed\":false}\n";
            throw std::runtime_error("Attention exceeds bounded workspace");
        }
        if (bounded) submissions=ernie::attention_internal_submissions(net);
    }
    if (actual.w!=output_width||actual.h!=queries||actual.c!=heads||actual.elempack!=1||actual.elemsize!=4u)
        throw std::runtime_error("Attention layout");
    if (actual.w!=unsliced.w||actual.h!=unsliced.h||actual.c!=unsliced.c||
        actual.elempack!=unsliced.elempack||actual.elemsize!=unsliced.elemsize)
        throw std::runtime_error("Sliced and unsliced attention layout differs");
    double maximum=0., error=0., energy=0.;
    for (int h=0; h<heads; ++h)
        for (int i=0; i<queries; ++i)
            for (int d=0; d<output_width; ++d)
            {
                const double a=actual.channel(h).row(i)[d], b=expected.channel(h).row(i)[d];
                if (!std::isfinite(a)) throw std::runtime_error("Nonfinite attention");
                if (actual.channel(h).row(i)[d]!=unsliced.channel(h).row(i)[d])
                    throw std::runtime_error("Chunking changed FP32 arithmetic");
                maximum=std::max(maximum,std::abs(a-b));error+=(a-b)*(a-b);energy+=b*b;
            }
    const double nrmse=std::sqrt(error/energy);
    const bool passed=maximum<=3e-6&&nrmse<=5e-7&&workspace.largest<=300*1024&&submissions==uint64_t((queries-1)/query_rows);
    std::cout << std::setprecision(12) << "{\"case\":\"" << name << "\",\"queries\":" << queries << ",\"keys\":" << keys
              << ",\"query_rows\":" << query_rows << ",\"kv_groups\":" << groups << ",\"output_width\":" << output_width << ",\"mask_channels\":" << mask_channels
              << ",\"largest_workspace_request\":" << workspace.largest << ",\"internal_submissions\":" << submissions
              << ",\"matches_unsliced_exactly\":true,\"max_abs\":" << maximum
              << ",\"nrmse\":" << nrmse << ",\"passed\":" << (passed?"true":"false") << "}\n";
    return passed;
}
#endif

int main()
{
#if !NCNN_VULKAN
    return 77;
#else
    int status=1;
    try
    {
        ncnn::create_gpu_instance();
        if (ncnn::get_gpu_count()<1) { ncnn::destroy_gpu_instance(); return 77; }
        bool passed=true;
        for (int rows : {128, 64, 32, 16})
        {
            passed=run_case("shared-mask",257,2,128,1,rows)&&passed;
            passed=run_case("per-head-mask",193,2,64,2,rows)&&passed;
            passed=run_case("grouped-query-mask",193,1,64,2,rows)&&passed;
            passed=run_case("grouped-query-no-mask",193,1,64,0,rows)&&passed;
        }
        status=passed?0:1;
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
    ncnn::destroy_gpu_instance();return status;
#endif
}

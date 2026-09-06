// SPDX-License-Identifier: MIT
#include "ernie_attention.h"
#include "layer.h"
#include <algorithm>
#include <atomic>
#include <memory>
#if NCNN_VULKAN
#include "vulkan/sdpa_vulkan.h"
#include "vulkan/softmax_vulkan.h"
#include "sdpa_shader.h"
#endif

namespace ernie {
namespace {
#if NCNN_VULKAN
const char* rows_shader = R"glsl(
#version 450
layout(binding=0) readonly buffer source_buffer { float source_data[]; };
layout(binding=1) writeonly buffer target_buffer { float target_data[]; };
layout(push_constant) uniform parameter {
    int width; int rows; int channels; int source_row; int target_row;
    int source_stride; int target_stride;
} p;
void main() {
    int i=int(gl_GlobalInvocationID.x), c=int(gl_GlobalInvocationID.z);
    if (i>=p.width*p.rows || c>=p.channels) return;
    target_data[c*p.target_stride+p.target_row*p.width+i] =
        source_data[c*p.source_stride+p.source_row*p.width+i];
}
)glsl";

class AccurateSDPA final : public ncnn::SDPA_vulkan
{
public:
    explicit AccurateSDPA(bool bounded) : bounded_workspace(bounded) {}
    int create_pipeline(const ncnn::Option& opt) override
    {
        const int rc = ncnn::SDPA_vulkan::create_pipeline(opt);
        if (rc || opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
            return rc;
        // Correct both long FP32 reductions:
        // softmax denominator and A @ V. Cache allocation/append, Q @ K and
        // all shape handling remain pinned ncnn. Low-storage flash is unchanged.
        std::vector<uint32_t> spirv;
        const int compiled = ncnn::compile_spirv_module(ernie_sdpa_shader, opt, spirv);
        if (compiled) return compiled;
        std::vector<ncnn::vk_specialization_type> specs(13);
        specs[1].f = 1.f;
        auto replacement = std::make_unique<ncnn::Pipeline>(vkdev);
        replacement->set_local_size_xyz(8, 8, 1);
        const int created = replacement->create(spirv.data(), spirv.size()*sizeof(uint32_t), specs);
        if (created) return created;
        delete pipeline_sdpa_qkv_cross;
        pipeline_sdpa_qkv_cross = replacement.release();
        spirv.clear();
        const int sum_compiled = ncnn::compile_spirv_module(ernie_softmax_sum_shader, opt, spirv);
        if (sum_compiled) return sum_compiled;
        specs.assign(13, {});
        specs[0].i = -1;
        replacement = std::make_unique<ncnn::Pipeline>(vkdev);
        replacement->set_optimal_local_size_xyz(4, 4, 4);
        const int sum_created = replacement->create(spirv.data(), spirv.size()*sizeof(uint32_t), specs);
        if (sum_created) return sum_created;
        auto* softmax = static_cast<ncnn::Softmax_vulkan*>(qk_softmax);
        delete softmax->pipeline_softmax_reduce_sum;
        softmax->pipeline_softmax_reduce_sum = replacement.release();
        if (bounded_workspace)
        {
            spirv.clear();
            const int compiled_rows = ncnn::compile_spirv_module(rows_shader, opt, spirv);
            if (compiled_rows) return compiled_rows;
            copy_rows = std::make_unique<ncnn::Pipeline>(vkdev);
            copy_rows->set_local_size_xyz(64, 1, 1);
            const int created_rows = copy_rows->create(spirv.data(), spirv.size()*sizeof(uint32_t), {});
            if (created_rows) return created_rows;
            if (copy_rows->shader_info().push_constant_count != 7) return -1;
        }
        return 0;
    }
    int destroy_pipeline(const ncnn::Option& opt) override
    {
        copy_rows.reset();
        return ncnn::SDPA_vulkan::destroy_pipeline(opt);
    }
    void rows(const ncnn::VkMat& src, ncnn::VkMat& dst, int from, int to, int count,
              ncnn::VkCompute& cmd) const
    {
        std::vector<ncnn::vk_constant_type> constants(7);
        constants[0].i=src.w; constants[1].i=count; constants[2].i=src.c;
        constants[3].i=from; constants[4].i=to;
        constants[5].i=src.cstep; constants[6].i=dst.cstep;
        ncnn::VkMat dispatch;
        dispatch.w=src.w*count; dispatch.h=1; dispatch.c=src.c;
        cmd.record_pipeline(copy_rows.get(), {src,dst}, constants, dispatch);
    }
    int forward(const std::vector<ncnn::VkMat>& in, std::vector<ncnn::VkMat>& out,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        const auto& q=in[0];
        // Preserve native autoregressive/cache and low-storage flash paths.
        // Static ERNIE DiT is batch one, pack1, with a full 2D attention mask.
        bool slice=copy_rows && !kv_cache && !use_flash_attention && q.h>128
                   && q.dims==3 && q.elempack==1 && q.elembits()==32;
#if NCNN_BATCH
        slice=slice && q.n==1;
#endif
        if (attn_mask)
            slice=slice && (in[3].dims==2 || in[3].dims==3)
                  && in[3].w==in[1].h && in[3].h==q.h
                  && in[3].elempack==1 && in[3].elembits()==32;
        if (!slice) return ncnn::SDPA_vulkan::forward(in,out,cmd,opt);
        out[0].create(in[2].w,q.h,q.c,4u,opt.blob_vkallocator);
        if (out[0].empty()) return -100;
        for (int first=0; first<q.h; first+=128)
        {
            const int count=std::min(128,q.h-first);
            auto part=in;
            part[0].create(q.w,count,q.c,4u,opt.workspace_vkallocator);
            if (part[0].empty()) return -100;
            rows(q,part[0],first,0,count,cmd);
            if (attn_mask)
            {
                part[3].create(in[3].w,count,in[3].c,4u,opt.workspace_vkallocator);
                if (part[3].empty()) return -100;
                rows(in[3],part[3],first,0,count,cmd);
            }
            std::vector<ncnn::VkMat> result(1);
            const int rc=ncnn::SDPA_vulkan::forward(part,result,cmd,opt);
            if (rc) return rc;
            rows(result[0],out[0],0,first,count,cmd);
            if (first+count<q.h)
            {
                // Complete scratch-buffer use before reusing bounded storage.
                // No activation leaves the GPU. The last chunk remains in the
                // caller's command, preserving subsequent graph operations.
                const int submitted=cmd.submit_and_wait();
                if (submitted) return submitted;
                internal_submissions.fetch_add(1,std::memory_order_relaxed);
                const int reset=cmd.reset();
                if (reset) return reset;
            }
        }
        return 0;
    }
    const bool bounded_workspace;
    std::unique_ptr<ncnn::Pipeline> copy_rows;
    mutable std::atomic<uint64_t> internal_submissions{0};
};
#endif

class ErnieSDPA final : public ncnn::Layer
{
public:
    explicit ErnieSDPA(bool bounded_workspace) : cpu(ncnn::create_layer_cpu("SDPA"))
    {
        one_blob_only = false;
#if NCNN_VULKAN
        support_vulkan = true;
        gpu = std::make_unique<AccurateSDPA>(bounded_workspace);
#endif
    }
    int load_param(const ncnn::ParamDict& params) override
    {
        if (!cpu) return -1;
        int rc = cpu->load_param(params);
#if NCNN_VULKAN
        if (!rc) rc = gpu->load_param(params);
#endif
        return rc;
    }
    int create_pipeline(const ncnn::Option& opt) override
    {
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            gpu->vkdev = vkdev;
            pipeline_status = gpu->create_pipeline(opt);
            return pipeline_status;
        }
#endif
        pipeline_status = cpu->create_pipeline(opt);
        return pipeline_status;
    }
    int destroy_pipeline(const ncnn::Option& opt) override
    {
        pipeline_status = -1;
        int rc = cpu->destroy_pipeline(opt);
#if NCNN_VULKAN
        gpu->destroy_pipeline(opt);
#endif
        return rc;
    }
    int forward(const std::vector<ncnn::Mat>& in, std::vector<ncnn::Mat>& out, const ncnn::Option& opt) const override
    {
        return pipeline_status ? pipeline_status : cpu->forward(in, out, opt);
    }
#if NCNN_VULKAN
    int forward(const std::vector<ncnn::VkMat>& in, std::vector<ncnn::VkMat>& out,
                ncnn::VkCompute& command, const ncnn::Option& opt) const override
    {
        return pipeline_status ? pipeline_status : gpu->forward(in, out, command, opt);
    }
    std::unique_ptr<AccurateSDPA> gpu;
#endif
    std::unique_ptr<ncnn::Layer> cpu;
    int pipeline_status = -1;
};
ncnn::Layer* ErnieSDPA_layer_creator(void* userdata)
{
    return new ErnieSDPA(userdata != nullptr);
}
} // namespace
int register_attention(ncnn::Net& net, bool bounded_workspace)
{
    static int bounded_marker;
    return net.register_custom_layer("SDPA", ErnieSDPA_layer_creator, nullptr,
                                     bounded_workspace ? &bounded_marker : nullptr);
}
uint64_t attention_internal_submissions(const ncnn::Net& net)
{
    uint64_t count=0;
#if NCNN_VULKAN
    for (const auto* layer:net.layers())
        if (layer->type=="SDPA")
            count+=static_cast<const ErnieSDPA*>(layer)->gpu->internal_submissions.load(std::memory_order_relaxed);
#endif
    return count;
}
} // namespace ernie

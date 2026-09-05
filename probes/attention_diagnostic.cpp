// SPDX-License-Identifier: MIT
#include "attention_diagnostic.h"
#include "vulkan/sdpa_vulkan.h"

namespace {
// Diagnostic-only wrapper around the unmodified pinned ncnn implementation.
// Match its branch predicate, then delegate the entire computation to it.
class ObservedSDPA final : public ncnn::SDPA_vulkan
{
public:
    explicit ObservedSDPA(AttentionDiagnostic* destination) : data(destination) {}
    int forward(const std::vector<ncnn::VkMat>& in, std::vector<ncnn::VkMat>& out,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        const bool flash = use_flash_attention && in[0].w % 8 == 0 && in[2].w % 8 == 0 && in[2].w <= FA_coopmat_N * 8;
        const int rc = ncnn::SDPA_vulkan::forward(in, out, cmd, opt);
        if (rc == 0)
        {
            ++data->calls;
            data->flash_calls += flash;
            data->cooperative_calls += flash && use_cooperative_matrix;
            data->tokens = in[0].h;
            data->head_dim = in[0].w;
        }
        return rc;
    }
private:
    AttentionDiagnostic* data;
};

ncnn::Layer* create(void* data) { return new ObservedSDPA(static_cast<AttentionDiagnostic*>(data)); }
}

int register_attention_diagnostic(ncnn::Net& net, AttentionDiagnostic& data)
{
    return net.register_custom_layer("SDPA", create, nullptr, &data);
}

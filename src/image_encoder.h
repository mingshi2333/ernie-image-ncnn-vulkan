// SPDX-License-Identifier: MIT
#pragma once
#include "model_package.h"
#include <mat.h>
#include "../include/ernie/pipeline.h"

namespace ernie
{
struct VaeEncoding
{
    ncnn::Mat mean;
    ncnn::Mat packed;
    ncnn::Mat normalized;
};

// Execute a separately authenticated full encoder graph. ComponentFiles carries
// graph text and the immutable weight path; package/source identity is verified
// by the caller before this numerical component is entered.
VaeEncoding encode_vae(const ComponentFiles &component, const RgbImage &rgb,
                       const ncnn::Option &option);
}

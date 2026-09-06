// SPDX-License-Identifier: MIT
#pragma once

namespace ernie
{
// Shares the process-global ncnn instance between this library's generation and
// diagnosis calls. A nested or concurrent diagnosis cannot destroy a live user.
// This guards instance lifetime, not concurrent inference memory consumption.
// An existing external instance is borrowed; its owner must retain it until all
// library calls finish. The library never destroys a borrowed instance.
class GpuContext
{
  public:
    explicit GpuContext(bool enabled, int requested_index = -1, bool require_device = true);
    ~GpuContext();
    GpuContext(const GpuContext &) = delete;
    GpuContext &operator=(const GpuContext &) = delete;

  private:
    bool enabled_ = false;
};
} // namespace ernie

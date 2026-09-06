// SPDX-License-Identifier: MIT
#pragma once
#include <string>

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
    // A diagnostic may observe a missing driver without acquiring an instance.
    // Do not call ncnn's creating queries when this is false.
    bool available() const noexcept { return enabled_; }
    const std::string &initialization_error() const noexcept { return initialization_error_; }

  private:
    bool enabled_ = false;
    std::string initialization_error_;
};
} // namespace ernie

// SPDX-License-Identifier: MIT
#pragma once
#include "net.h"
#include <vector>

namespace ernie
{
// The referenced CPU FP32 Nets must outlive the session. Model weights may be
// shared by sessions; their opaque KV caches and allocators are always separate.
class PeSession
{
  public:
    explicit PeSession(std::vector<const ncnn::Net *> blocks, int capacity);
    PeSession(const PeSession &) = delete;
    PeSession &operator=(const PeSession &) = delete;
    ncnn::Mat step(const ncnn::Mat &embedded, const ncnn::Mat &cos, const ncnn::Mat &sin);
    void reset();
    int position() const
    {
        return position_;
    }
    size_t cache_buffer_changes() const
    {
        return buffer_changes_;
    }

  private:
    // Destruction order releases every cache handle before the allocator.
    ncnn::UnlockedPoolAllocator cache_allocator_;
    std::vector<const ncnn::Net *> blocks_;
    std::vector<ncnn::Mat> keys_, values_;
    int capacity_, position_ = 0;
    size_t buffer_changes_ = 0;
    bool valid_ = true;
};
void load_pe_block(ncnn::Net &net, const std::string &directory);
} // namespace ernie

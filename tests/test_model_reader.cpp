// SPDX-License-Identifier: MIT
#include "datareader.h"
#include "modelbin.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
std::atomic<std::size_t> largest_new{0};
std::atomic<bool> observe_new{false};
}

// Observe the real upstream std::vector allocation, separately from Mat's
// aligned allocator. Sampling RSS cannot prove the vector's requested size.
void* operator new(std::size_t bytes)
{
    if (void* value = std::malloc(bytes ? bytes : 1))
    {
        if (observe_new.load(std::memory_order_relaxed))
        {
            auto old = largest_new.load(std::memory_order_relaxed);
            while (old < bytes && !largest_new.compare_exchange_weak(old, bytes, std::memory_order_relaxed)) {}
        }
        return value;
    }
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

namespace {
void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
const std::uint16_t fp16[] = {0x0000, 0x8000, 0x3c00, 0xc100, 0x0001};
const std::uint16_t bf16[] = {0x0000, 0x8000, 0x3f80, 0xc020, 0x0001};
const std::uint32_t expected[] = {0, 0x80000000u, 0x3f800000u, 0xc0200000u};

class GeneratedReader final : public ncnn::DataReader
{
public:
    GeneratedReader(int count, bool bfloat, int truncate = 0)
        : count_(count), bfloat_(bfloat), payload_((std::size_t(count) * 2 + 3) / 4 * 4),
          size_(4 + payload_ - truncate) {}
    std::size_t read(void* output, std::size_t bytes) const override
    {
        if (position_ == 0)
        {
            largest_new.store(0);
            observe_new.store(true);
        }
        else if (position_ == 4)
        {
            allocation_at_payload = largest_new.load();
            observe_new.store(false);
        }
        auto* data = static_cast<unsigned char*>(output);
        const auto actual = std::min(bytes, size_ - position_);
        const std::uint32_t tag = bfloat_ ? 0x01348b83 : 0x01306b47;
        for (std::size_t i = 0; i < actual; ++i)
        {
            const auto offset = position_ + i;
            if (offset < 4) data[i] = (tag >> (8 * offset)) & 255;
            else
            {
                const auto word = (offset - 4) / 2;
                const auto value = word < std::size_t(count_) ? (bfloat_ ? bf16 : fp16)[word % 5] : 0x7fff;
                data[i] = (value >> (8 * ((offset - 4) % 2))) & 255;
            }
        }
        position_ += actual;
        return actual;
    }
    std::size_t payload() const { return payload_; }
    std::size_t consumed() const { return position_; }
    mutable std::size_t allocation_at_payload = 0;
private:
    int count_;
    bool bfloat_;
    std::size_t payload_, size_;
    mutable std::size_t position_ = 0;
};

void verify_values(const ncnn::Mat& values, int count, bool bfloat)
{
    require(!values.empty() && values.w == count && values.elemsize == 4u, "Wrong decoded weight shape/type");
    for (int i = 0; i < count; ++i)
    {
        std::uint32_t bits;
        std::memcpy(&bits, static_cast<const float*>(values) + i, 4);
        const auto reference = i % 5 == 4 ? (bfloat ? 0x00010000u : 0x33800000u) : expected[i % 5];
        require(bits == reference, "Decoded weight bits changed, including signed zero/subnormals");
    }
}

std::size_t vector_allocation(std::size_t elements)
{
    // Measure the same empty-vector resize on this standard library. MSVC
    // adds alignment metadata to large allocations, beyond the element bytes.
    largest_new.store(0);
    observe_new.store(true);
    std::vector<unsigned short> control;
    control.resize(elements);
    observe_new.store(false);
    const auto bytes = largest_new.load();
    require(bytes >= elements * sizeof(unsigned short), "Missing vector allocation control");
    return bytes;
}

void exercise(int count, bool bfloat)
{
    GeneratedReader reader(count, bfloat);
    const auto values = ncnn::ModelBinFromDataReader(reader).load(count, 0);
    observe_new.store(false);
    verify_values(values, count, bfloat);
    require(reader.consumed() == reader.payload() + 4, "Incorrect aligned input consumption");
    const auto multiplier = ERNIE_COMPACT_MODEL_READER_TEST ? 1 : sizeof(unsigned short);
    const auto vector_bytes = reader.payload() * multiplier;
    const auto expected_allocation = vector_allocation(vector_bytes / sizeof(unsigned short));
    require(reader.allocation_at_payload == expected_allocation, "Unexpected temporary vector allocation");

    // A reference-capable reader takes the same mapped input branch as ncnn.
    std::vector<unsigned char> encoded(4 + reader.payload());
    GeneratedReader source(count, bfloat);
    require(source.read(encoded.data(), encoded.size()) == encoded.size(), "Fixture encoding failed");
    observe_new.store(false);
    const unsigned char* cursor = encoded.data();
    ncnn::DataReaderFromMemory mapped(cursor);
    verify_values(ncnn::ModelBinFromDataReader(mapped).load(count, 0), count, bfloat);
    require(cursor == encoded.data() + encoded.size(), "Mapped read lost alignment");
    std::cout << (bfloat ? "BF16" : "FP16") << " count=" << count << " payload=" << reader.payload()
              << " vector_bytes=" << reader.allocation_at_payload
              << " vector_payload_bytes=" << vector_bytes << " bit_exact=true\n";
}
} // namespace

int main()
{
    try
    {
        for (bool bfloat : {false, true})
        {
            for (int count : {1, 2, 3, 17, 2049, 65537, 4194305}) exercise(count, bfloat);
            GeneratedReader truncated(17, bfloat, 1);
            require(ncnn::ModelBinFromDataReader(truncated).load(17, 0).empty(), "Truncated aligned payload was accepted");
            observe_new.store(false);
            GeneratedReader missing_tag(1, bfloat, 6);
            require(ncnn::ModelBinFromDataReader(missing_tag).load(1, 0).empty(), "Truncated tag was accepted");
            observe_new.store(false);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        observe_new.store(false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}

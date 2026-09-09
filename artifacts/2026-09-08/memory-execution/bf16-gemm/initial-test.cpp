#include "allocator.h"
#include "command.h"
#include "gpu.h"
#include "net.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Shape
{
    const char* name;
    int m, n, k;
    int transpose_a, transpose_b, bias;
    int transpose_output, output_n1m, output_pack;
};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

float input_value(int row, int inner)
{
    return float((row * 13 + inner * 7) % 17 - 3) * (row % 3 == 0 ? -0.125f : 0.125f);
}

float weight_value(int inner, int column)
{
    return float((inner * 5 + column * 11) % 9 - 2) * 0.125f;
}

float bias_value(const Shape& shape, int row, int column)
{
    if (shape.bias == -1) return 0.f;
    return float(((shape.bias == 3 ? row * 3 : 0) + column) % 5 - 2) * 0.125f;
}

float rounded_bf16(double value)
{
    // Independent round-to-nearest-even oracle. All operands are exact BF16
    // multiples of 1/8; every FP32/FP64 product and partial sum is exact, so
    // reduction order cannot hide a matrix/layout error behind a tolerance.
    float result = static_cast<float>(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &result, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::string graph_for(const Shape& shape)
{
    std::ostringstream graph;
    graph << "7767517\n2 2\nInput input 0 1 in\n"
          << "Gemm projection 1 1 in out 0=1 1=1"
          << " 2=" << shape.transpose_a << " 3=" << shape.transpose_b
          << " 4=0 5=1 6=1 7=" << shape.m << " 8=" << shape.n
          << " 9=" << shape.k << " 10=" << shape.bias
          << " 11=" << shape.output_n1m << " 12=" << shape.output_pack
          << " 14=" << shape.transpose_output << '\n';
    return graph.str();
}

std::vector<float> weights_for(const Shape& shape)
{
    // ModelBin type 0 prefixes each FP32 weight tensor with a zero tag.
    std::vector<float> weights(1, 0.f);
    const int rows = shape.transpose_b ? shape.n : shape.k;
    const int columns = shape.transpose_b ? shape.k : shape.n;
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            weights.push_back(weight_value(shape.transpose_b ? column : row,
                                           shape.transpose_b ? row : column));
    if (shape.bias != -1)
    {
        weights.push_back(0.f);
        for (int row = 0; row < (shape.bias == 3 ? shape.m : 1); ++row)
            for (int column = 0; column < shape.n; ++column)
                weights.push_back(bias_value(shape, row, column));
    }
    return weights;
}

std::vector<float> run(const Shape& shape, const ncnn::VulkanDevice* device,
                       bool cooperative)
{
    std::cout << "CASE " << shape.name << " M=" << shape.m << " N=" << shape.n
              << " K=" << shape.k << " transA=" << shape.transpose_a
              << " transB=" << shape.transpose_b << " bias=" << shape.bias
              << " output_transpose=" << shape.transpose_output
              << " output_N1M=" << shape.output_n1m
              << " output_pack=" << shape.output_pack
              << " cooperative_requested=" << cooperative << std::endl;

    ncnn::VkBlobAllocator blobs(device, 0);
    ncnn::VkStagingAllocator staging(device);
    ncnn::Net net;
    net.set_vulkan_device(device);
    net.opt.num_threads = 2;
    net.opt.use_vulkan_compute = true;
    net.opt.use_bf16_storage = true;
    net.opt.use_bf16_packed = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_cooperative_matrix = cooperative;
    require(net.load_param_mem(graph_for(shape).c_str()) == 0, "load Gemm graph");
    const auto weights = weights_for(shape);
    require(net.load_model(reinterpret_cast<const unsigned char*>(weights.data())) ==
                weights.size() * sizeof(float), "load Gemm weights");

    ncnn::Mat input(shape.transpose_a ? shape.m : shape.k,
                    shape.transpose_a ? shape.k : shape.m);
    for (int row = 0; row < input.h; ++row)
        for (int column = 0; column < input.w; ++column)
            input.row(row)[column] = input_value(shape.transpose_a ? column : row,
                                                shape.transpose_a ? row : column);

    ncnn::Option option = net.opt;
    option.blob_vkallocator = option.workspace_vkallocator = &blobs;
    option.staging_vkallocator = &staging;
    ncnn::Mat output;
    {
        ncnn::VkMat gpu_input, gpu_output;
        ncnn::VkCompute command(device);
        command.record_upload(input, gpu_input, option);
        require(!gpu_input.empty() && gpu_input.elembits() == 16, "input must use native BF16 storage");
        auto extractor = net.create_extractor();
        extractor.set_blob_vkallocator(&blobs);
        extractor.set_workspace_vkallocator(&blobs);
        extractor.set_staging_vkallocator(&staging);
        require(extractor.input("in", gpu_input) == 0, "set Gemm input");
        require(extractor.extract("out", gpu_output, command) == 0, "execute Vulkan Gemm");
        require(!gpu_output.empty() && gpu_output.elembits() == 16,
                "output must retain native BF16 storage");
        require(gpu_output.elempack == shape.output_pack, "Gemm output packing differs");
        ncnn::Option download = option;
        download.use_packing_layout = false;
        command.record_download(gpu_output, output, download);
        require(command.submit_and_wait() == 0, "submit Vulkan Gemm");
    }

    const int width = shape.transpose_output ? shape.m : shape.n;
    const int rows = shape.transpose_output ? shape.n : shape.m;
    require(!output.empty() && output.elempack == 1 && output.elembits() == 32 &&
                output.w == width && output.dims == (shape.output_n1m ? 3 : 2) &&
                (shape.output_n1m ? output.h == 1 && output.c == rows : output.h == rows),
            "downloaded Gemm layout differs");
    std::vector<float> result(static_cast<size_t>(shape.m) * shape.n);
    for (int row = 0; row < shape.m; ++row)
        for (int column = 0; column < shape.n; ++column)
        {
            const int y = shape.transpose_output ? column : row;
            const int x = shape.transpose_output ? row : column;
            const float actual = shape.output_n1m ? output.channel(y).row(0)[x] : output.row(y)[x];
            double sum = bias_value(shape, row, column);
            for (int inner = 0; inner < shape.k; ++inner)
                sum += double(input_value(row, inner)) * double(weight_value(inner, column));
            const float expected = rounded_bf16(sum);
            if (!std::isfinite(actual) || actual != expected)
            {
                std::ostringstream error;
                error << shape.name << " cooperative=" << cooperative << " at ["
                      << row << ',' << column << "]: " << actual << " != " << expected;
                throw std::runtime_error(error.str());
            }
            result[static_cast<size_t>(row) * shape.n + column] = actual;
        }
    return result;
}

} // namespace

int main()
{
    try
    {
        if (ncnn::create_gpu_instance() || ncnn::get_gpu_count() == 0)
        {
            std::cout << "SKIP: no Vulkan device\n";
            return 77;
        }
        const auto* device = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
        if (!device->info.support_bf16_storage())
        {
            std::cout << "SKIP: Vulkan device does not support BF16 storage\n";
            return 77;
        }
        std::cout << "BF16 storage supported; BF16 cooperative capability="
                  << device->info.support_bf16_cooperative_matrix() << std::endl;
        const Shape cases[] = {
            {"projection", 32, 48, 64, 0, 1, -1, 0, 0, 4},
            {"single_row_bias", 1, 48, 32, 0, 1, 4, 0, 0, 1},
            {"odd_transpose", 17, 19, 32, 1, 0, 4, 1, 1, 1},
            {"matrix_bias", 32, 48, 64, 1, 1, 3, 1, 1, 4},
        };
        for (const auto& shape : cases)
        {
            const auto requested = run(shape, device, true);
            const auto disabled = run(shape, device, false);
            require(requested == disabled, std::string(shape.name) + " cooperative ON/OFF differs");
        }
        std::cout << "BF16 Gemm: 4 shapes, cooperative ON/OFF and independent FP64 oracle exact\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

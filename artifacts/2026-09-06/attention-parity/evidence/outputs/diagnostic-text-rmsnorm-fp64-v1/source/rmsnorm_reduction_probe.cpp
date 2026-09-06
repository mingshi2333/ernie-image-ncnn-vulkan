// SPDX-License-Identifier: MIT
#include "ernie_rmsnorm.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Unequal energy across a long feature axis exposes loss of small squares.
// Exercise both packed and unpacked rows; the oracle is independent FP64.
int main()
{
    try
    {
        bool passed = true;
        for (int width : {128, 3072, 4096})
            for (int rows : {1, 9, 64})
                for (bool packed : {false, true})
                {
                    ncnn::Net net;
                    net.opt.num_threads = 4;
                    net.opt.use_vulkan_compute = false;
                    net.opt.use_packing_layout = packed;
                    net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_bf16_storage = false;
                    if (ernie::register_rmsnorm(net)) throw std::runtime_error("Register norm");
                    const std::string graph = "7767517\n2 2\nInput input 0 1 in0\nRMSNorm norm 1 1 in0 out0 0="
                        + std::to_string(width) + " 1=1e-5 2=1\n";
                    std::vector<float> weights(width);
                    for (int i = 0; i < width; ++i) weights[i] = .5f + (i%8)*.0625f;
                    if (net.load_param_mem(graph.c_str()) || net.load_model(reinterpret_cast<const unsigned char*>(weights.data())) < 0)
                        throw std::runtime_error("Load norm");
                    ncnn::Mat input(width, rows), output;
                    for (int r = 0; r < rows; ++r)
                        for (int x = 0; x < width; ++x)
                            input.row(r)[x] = x == r%width ? 512.f : .125f;
                    auto ex = net.create_extractor();
                    if (ex.input("in0", input) || ex.extract("out0", output)) throw std::runtime_error("Run norm");
                    if (output.w != width || output.h != rows || output.elempack != 1 || output.elemsize != 4u)
                        throw std::runtime_error("Norm shape");
                    double maximum = 0., reference_maximum = 0., error = 0., energy = 0.;
                    for (int r = 0; r < rows; ++r)
                    {
                        double squares = 0.;
                        for (int x = 0; x < width; ++x) squares += double(input.row(r)[x])*input.row(r)[x];
                        const double inverse = 1./std::sqrt(squares/width+double(1e-5f));
                        for (int x = 0; x < width; ++x)
                        {
                            const double expected = input.row(r)[x]*inverse*weights[x];
                            const double actual = output.row(r)[x];
                            if (!std::isfinite(actual)) throw std::runtime_error("Nonfinite norm");
                            maximum = std::max(maximum, std::abs(actual-expected));
                            reference_maximum = std::max(reference_maximum, std::abs(expected));
                            error += (actual-expected)*(actual-expected); energy += expected*expected;
                        }
                    }
                    const double nrmse = std::sqrt(error/energy), limit = 2e-6+2e-7*reference_maximum;
                    const bool ok = maximum <= limit && nrmse <= 2e-7;
                    passed &= ok;
                    std::cout << std::setprecision(12) << "{\"width\":" << width << ",\"rows\":" << rows
                              << ",\"packing\":" << (packed ? "true" : "false") << ",\"max_abs\":" << maximum
                              << ",\"max_abs_limit\":" << limit << ",\"nrmse\":" << nrmse
                              << ",\"nrmse_limit\":2e-7,\"passed\":" << (ok ? "true" : "false") << "}\n";
                }
        return passed ? 0 : 1;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

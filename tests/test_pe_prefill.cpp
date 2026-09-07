// SPDX-License-Identifier: MIT
#include "layer.h"
#include "pe_graph.h"
#include "pe_session.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{
void require(bool ok, const char *what)
{
    if (!ok)
        throw std::runtime_error(what);
}
template <class F> void rejects(F action)
{
    bool rejected = false;
    try
    {
        action();
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    require(rejected, "Expected rejection");
}
ncnn::Option cpu()
{
    ncnn::Option o;
    o.num_threads = 2;
    o.use_vulkan_compute = false;
    o.use_fp16_storage = o.use_fp16_packed = o.use_fp16_arithmetic = false;
    o.use_bf16_storage = o.use_bf16_packed = false;
    return o;
}
// Test adapters reduce the real contract to a two-dimensional attention head.
// The ncnn SDPA between them is the actual native opaque cache implementation.
class Prepare final : public ncnn::Layer
{
  public:
    int forward(const std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out,
                const ncnn::Option &) const override
    {
        for (const auto &m : in)
            if (m.dims != 2 || !m.refcount)
                return -1;
        for (auto &m : out)
            m.create(2, in[0].h, 1);
        for (int r = 0; r < in[0].h; ++r)
            for (int j = 0; j < 2; ++j)
            {
                out[0].row(r)[j] = in[0].row(r)[j];
                out[1].row(r)[j] = in[0].row(r)[j] * .5f;
                out[2].row(r)[j] = in[0].row(r)[j] + float(j);
            }
        return 0;
    }
};
class Expand final : public ncnn::Layer
{
  public:
    Expand()
    {
        one_blob_only = true;
    }
    int forward(const ncnn::Mat &in, ncnn::Mat &out, const ncnn::Option &) const override
    {
        out.create(3072, in.h);
        for (int r = 0; r < in.h; ++r)
            for (int j = 0; j < 3072; ++j)
                out.row(r)[j] = in.row(r)[j % 2];
        return 0;
    }
};
// Lightweight state fixture: cache handles carry only allocator/length metadata.
// No native opaque cache is inspected/copied as ordinary tensor values.
class State final : public ncnn::Layer
{
    int failure = 0;

  public:
    int load_param(const ncnn::ParamDict &pd) override
    {
        failure = pd.get(0, 0);
        return 0;
    }
    int forward(const std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out,
                const ncnn::Option &opt) const override
    {
        const int past = in[4].empty() ? 0 : in[4].h, rows = in[0].h;
        for (int i = 0; i < 3; ++i)
            if (in[i].dims != 2 || !in[i].refcount)
                return -1;
        if (in[3].w != past + rows || in[3].h != rows)
            return -1;
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < past + rows; ++j)
                if (in[3].row(r)[j] != (j <= past + r ? 0.f : std::numeric_limits<float>::lowest()))
                    return -1;
        out[0] = in[0].clone();
        for (int i = 1; i <= 2; ++i)
        {
            if (in[i + 3].empty())
                out[i].create(1, opt.kvcache_max_seqlen_hint, 4u, opt.kvcache_allocator);
            else
                out[i] = in[i + 3];
            out[i].h = past + rows;
        }
        // Exercise alias ownership: these fixture inputs are intentionally mutated.
        for (int i = 0; i < 3; ++i)
            static_cast<float *>(in[i].data)[0] += 1.f;
        if (failure == 1 || (failure == 8 && past > 0))
            return -1;
        if (failure == 2)
            out[0] = out[0].reshape(1536, rows * 2);
        if (failure == 3)
            out[0][0] = std::numeric_limits<float>::infinity();
        if (failure == 4)
            out[1].h = past + rows + 1;
        if (failure == 5)
            out[2].release();
        if (failure == 6)
            out[1] = ncnn::Mat(1, past + rows);
        if (failure == 7)
            out[2].h = past + rows + 1;
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Prepare)
DEFINE_LAYER_CREATOR(Expand)
DEFINE_LAYER_CREATOR(State)
void load(ncnn::Net &net, int state = -1, const std::string &missing = {})
{
    net.opt = cpu();
    net.register_custom_layer("Prepare", Prepare_layer_creator);
    net.register_custom_layer("Expand", Expand_layer_creator);
    net.register_custom_layer("State", State_layer_creator);
    const std::string inputs = "Input x 0 1 in0\nInput cos 0 1 in1\nInput sin 0 1 in2\n"
                               "Input mask 0 1 in3\nInput cache 0 2 past_k past_v\n";
    std::string graph = state < 0
                            ? "7767517\n8 13\n" + inputs +
                                  "Prepare prepare 3 3 in0 in1 in2 q k v\n"
                                  "SDPA attention 6 3 q k v in3 past_k past_v attended out_k out_v 5=1 7=1\n"
                                  "Expand expand 1 1 attended out0\n"
                            : "7767517\n6 9\n" + inputs +
                                  "State state 6 3 in0 in1 in2 in3 past_k past_v out0 out_k out_v 0=" +
                                  std::to_string(state) + "\n";
    if (!missing.empty())
        graph.replace(graph.find(missing), missing.size(), "missing");
    require(net.load_param_mem(graph.c_str()) == 0, "Synthetic graph load failed");
    static const unsigned char empty_weights[4] = {};
    net.load_model(empty_weights);
}
struct Inputs
{
    ncnn::Mat x, cos, sin;
    explicit Inputs(int rows) : x(3072, rows), cos(128, rows), sin(128, rows)
    {
        cos.fill(1.f);
        sin.fill(0.f);
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < 3072; ++j)
                x.row(r)[j] = float((r * 13 + j * 7) % 31 - 15) / 32.f;
    }
    ncnn::Mat append(ernie::PeSession &s, int first, int count) const
    {
        return s.append_chunk(x.row_range(first, count), cos.row_range(first, count),
                              sin.row_range(first, count));
    }
};
void graph_tests()
{
    std::ifstream file(ERNIE_PE_GRAPH_FIXTURE, std::ios::binary);
    const std::string original((std::istreambuf_iterator<char>(file)), {});
    require(!original.empty(), "Missing PE graph fixture");
    const auto derived = ernie::pe_chunk_graph(original);
    // Independent token diff: only the four explicitly enumerated fields differ.
    std::istringstream old_lines(original), new_lines(derived);
    std::string a, b;
    int changes = 0;
    while (std::getline(old_lines, a))
    {
        require(bool(std::getline(new_lines, b)), "Derived graph lost a line");
        if (a == b)
            continue;
        std::istringstream old_tokens(a), new_tokens(b);
        std::string type, name, token, newer;
        old_tokens >> type >> name;
        new_tokens >> token >> newer;
        require(type == "Reshape" && token == type && newer == name, "Changed layer identity");
        require(name == "reshape_10" || name == "reshape_11" || name == "reshape_12" || name == "reshape_13",
                "Changed unrelated layer");
        int fields = 0;
        while (old_tokens >> token)
        {
            require(bool(new_tokens >> newer), "Lost token");
            if (token == newer)
                continue;
            require(token == (name == "reshape_13" ? "1=1" : "2=1") &&
                        newer == (name == "reshape_13" ? "1=-1" : "2=-1"),
                    "Changed wrong dimension");
            ++fields;
        }
        require(!(new_tokens >> newer) && fields == 1, "Unexpected extra dimension edits");
        ++changes;
    }
    require(!std::getline(new_lines, b) && changes == 4, "Expected exactly four sequence changes");
    rejects([&] { ernie::pe_chunk_graph(""); });
    rejects([&] { ernie::pe_chunk_graph(derived); });
    rejects([&] { ernie::pe_chunk_graph(original + "Input extra 0 1 extra\n"); });
    rejects([&] { ernie::pe_chunk_graph(original + "Reshape reshape_10 1 1 10 11 0=128 1=32 2=1\n"); });
    for (const auto &pair : {std::pair<std::string, std::string>{"reshape_10", "reshape_11"},
                             {"59 79", "58 79"},
                             {"0=128", "0=127"},
                             {"7=1", "7=0"}})
    {
        auto bad = original;
        const auto at = bad.find(pair.first);
        require(at != std::string::npos, "Missing mutation target");
        bad.replace(at, pair.first.size(), pair.second);
        rejects([&] { ernie::pe_chunk_graph(bad); });
    }
}
void attention_tests()
{
    ncnn::Net net;
    load(net);
    Inputs in(17);
    const auto original = in.x.clone();
    for (const auto &plan : {std::vector<int>{17}, {8, 8, 1}, {5, 1, 11}, std::vector<int>(17, 1)})
    {
        ernie::PeSession s({&net}, 17), independent({&net}, 17);
        ncnn::Mat actual(3072, 17);
        int first = 0;
        for (int count : plan)
        {
            const auto out = in.append(s, first, count);
            const auto other = in.append(independent, first, count);
            require(!std::memcmp(out.data, other.data, size_t(count) * 3072 * 4),
                    "Independent session differs");
            std::memcpy(actual.row(first), out.data, size_t(count) * 3072 * 4);
            first += count;
            require(s.position() == first, "Incorrect position advance");
        }
        // Independent scalar full-causal oracle, directly from the fixture values.
        for (int r = 0; r < 17; ++r)
        {
            double total = 0, values[2] = {};
            for (int j = 0; j <= r; ++j)
            {
                double score = 0;
                for (int k = 0; k < 2; ++k)
                    score += double(in.x.row(r)[k]) * in.x.row(j)[k] * .5;
                const double weight = std::exp(score / std::sqrt(2.0));
                total += weight;
                for (int k = 0; k < 2; ++k)
                    values[k] += weight * (in.x.row(j)[k] + k);
            }
            for (int k = 0; k < 3072; ++k)
                require(std::abs(actual.row(r)[k] - values[k % 2] / total) < 2e-6,
                        "Scalar causal oracle differs");
        }
        require(s.cache_buffer_changes() == 2, "Opaque cache reallocated inside capacity");
        s.reset();
        first = 0;
        for (int count : plan)
        {
            const auto out = in.append(s, first, count);
            require(!std::memcmp(out.data, actual.row(first), size_t(count) * 3072 * 4),
                    "Reset replay differs");
            first += count;
        }
        rejects([&] { in.append(s, 0, 1); });
        require(s.position() == 17, "Capacity rejection changed position");
    }
    require(!std::memcmp(in.x.data, original.data, 17 * 3072 * 4), "Attention input changed");
}
void state_tests()
{
    ncnn::Net net;
    load(net, 0);
    Inputs in(33);
    ernie::PeSession s({&net}, 4096);
    const auto original = in.x.clone(), saved_cos = in.cos.clone(), saved_sin = in.sin.clone();
    in.append(s, 0, 32);
    require(s.position() == 32, "32-token boundary failed");
    // Prevalidation errors must preserve live caches and position.
    auto bad = [&](const ncnn::Mat &x, const ncnn::Mat &cos, const ncnn::Mat &sin)
    {
        rejects([&] { s.append_chunk(x, cos, sin); });
        require(s.position() == 32 && s.cache_buffer_changes() == 2, "Parameter error poisoned session");
    };
    bad(in.x, in.cos, in.sin);
    bad(ncnn::Mat(), in.cos, in.sin);
    bad(ncnn::Mat(3071, 1), in.cos.row_range(0, 1), in.sin.row_range(0, 1));
    bad(in.x.row_range(0, 2), in.cos.row_range(0, 1), in.sin.row_range(0, 2));
    bad(ncnn::Mat(3072, 1, 1), in.cos.row_range(0, 1), in.sin.row_range(0, 1));
    bad(ncnn::Mat(3072, 1, size_t(2)), in.cos.row_range(0, 1), in.sin.row_range(0, 1));
    bad(ncnn::Mat(3072, 1, size_t(16), 4), in.cos.row_range(0, 1), in.sin.row_range(0, 1));
    for (float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    {
        auto x = in.x.row_range(0, 1).clone();
        x[0] = invalid;
        bad(x, in.cos.row_range(0, 1), in.sin.row_range(0, 1));
        auto c = in.cos.row_range(0, 1).clone();
        c[127] = invalid;
        bad(in.x.row_range(0, 1), c, in.sin.row_range(0, 1));
        auto v = in.sin.row_range(0, 1).clone();
        v[127] = invalid;
        bad(in.x.row_range(0, 1), in.cos.row_range(0, 1), v);
    }
    rejects([&] { s.step(in.x.row_range(0, 2), in.cos.row_range(0, 2), in.sin.row_range(0, 2)); });
    for (int i = 1; i < 128; ++i)
        in.append(s, 0, 32);
    require(s.position() == 4096 && s.cache_buffer_changes() == 2, "4096 exact capacity fixture failed");
    rejects([&] { in.append(s, 0, 1); });
    s.reset();
    // External vector and legacy one-row channel views normalize to owned 2D.
    ncnn::Mat external(3072, in.x.data), external_cos(128, in.cos.data), external_sin(128, in.sin.data);
    s.step(external, external_cos, external_sin);
    s.step(in.x.row_range(0, 1).reshape(3072, 1, 1), in.cos.row_range(0, 1), in.sin.row_range(0, 1));
    require(!std::memcmp(in.x.data, original.data, 33 * 3072 * 4) &&
                !std::memcmp(in.cos.data, saved_cos.data, 33 * 128 * 4) &&
                !std::memcmp(in.sin.data, saved_sin.data, 33 * 128 * 4),
            "Owned normalization mutated external/view data");
    for (int failure = 1; failure <= 7; ++failure)
    {
        ncnn::Net faulty;
        load(faulty, failure);
        ernie::PeSession broken({&net, &faulty}, 64);
        rejects([&] { in.append(broken, 0, 2); });
        require(broken.position() == 0 && broken.cache_buffer_changes() == 0, "Failure did not clear state");
        rejects([&] { in.append(broken, 0, 1); });
        // Reload the fixture weights/graph after no extractor remains, then reset.
        faulty.clear();
        load(faulty, 0);
        rejects([&] { in.append(broken, 0, 1); });
        broken.reset();
        in.append(broken, 0, 1);
        require(broken.position() == 1 && broken.cache_buffer_changes() == 4,
                "Failed session reset did not recover");
    }
    {
        ncnn::Net faulty;
        load(faulty, 8);
        ernie::PeSession broken({&net, &faulty}, 64);
        in.append(broken, 0, 5);
        require(broken.position() == 5, "Prefix before injected cache failure failed");
        rejects([&] { in.append(broken, 5, 2); });
        require(broken.position() == 0 && broken.cache_buffer_changes() == 0,
                "Existing prefix survived nontransactional cache failure");
        rejects([&] { in.append(broken, 0, 1); });
        broken.reset();
        in.append(broken, 0, 1);
    }
    for (const auto *missing : {"out0", "out_k", "out_v"})
    {
        ncnn::Net faulty;
        load(faulty, 0, missing);
        ernie::PeSession broken({&net, &faulty}, 64);
        rejects([&] { in.append(broken, 0, 2); });
        require(broken.position() == 0 && broken.cache_buffer_changes() == 0,
                "Missing output did not clear all layers");
        faulty.clear();
        load(faulty, 0);
        rejects([&] { in.append(broken, 0, 1); });
        broken.reset();
        in.append(broken, 0, 1);
    }
    rejects([&] { ernie::PeSession empty({}, 1); });
    rejects([&] { ernie::PeSession too_many(std::vector<const ncnn::Net *>(27, &net), 1); });
    rejects([&] { ernie::PeSession zero({&net}, 0); });
    rejects([&] { ernie::PeSession too_large({&net}, 4097); });
    rejects([&] { ernie::PeSession null({nullptr}, 1); });
    ncnn::Net half;
    half.opt = cpu();
    half.opt.use_fp16_storage = true;
    rejects([&] { ernie::PeSession wrong({&half}, 1); });
    ernie::PeSession all_blocks(std::vector<const ncnn::Net *>(26, &net), 1);
    in.append(all_blocks, 0, 1);
    require(all_blocks.cache_buffer_changes() == 52, "26 block session cache contract failed");
}
} // namespace
int main()
{
    try
    {
        graph_tests();
        attention_tests();
        state_tests();
        std::cout << "PE graph, native causal attention, ownership, validation, failure/reset and4096 "
                     "capacity passed\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

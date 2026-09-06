// SPDX-License-Identifier: MIT
#include "ernie_text_down.h"
#include "text_encoder.h"
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace fs=std::filesystem;
void require(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F action)
{ bool rejected=false;try { action(); } catch (const std::exception&) { rejected=true; } require(rejected,"Missing rejection"); }
ncnn::Option options()
{
    ncnn::Option o;o.num_threads=2;o.use_vulkan_compute=false;
    o.use_fp16_storage=o.use_fp16_packed=o.use_fp16_arithmetic=o.use_bf16_storage=o.use_bf16_packed=false;
    return o;
}
std::string graph(const std::string& params)
{ return "7767517\n2 2\nInput in0 0 1 in0\nErnieTextDown down 1 1 in0 out0 "+params+"\n"; }
constexpr const char* good="0=3072 1=0 2=28311552";
void structural_weights(const fs::path& p,bool bf16)
{
    std::ofstream out(p,std::ios::binary);std::uint64_t offset=3072*4;
    auto matrix=[&](std::uint64_t count) {
        out.seekp(offset);const unsigned char fp32[4]={0,0,0,0}, b[4]={0x83,0x8b,0x34,0x01};
        out.write(reinterpret_cast<const char*>(bf16?b:fp32),4);offset+=4+count*(bf16?2:4);
    };
    matrix(4096ull*3072);matrix(1024ull*3072);matrix(1024ull*3072);matrix(3072ull*4096);
    offset+=3072*4;matrix(9216ull*3072);matrix(9216ull*3072);matrix(3072ull*9216);
    out.seekp(offset-1);out.put(0);out.close();require(bool(out),"Sparse weight fixture failed");
}
int main(int argc,char** argv)
{
    const fs::path temp=fs::temp_directory_path()/("ernie-text-down-"+std::to_string(getpid()));
    try
    {
        if (argc==2)
        {
            const auto start=std::chrono::steady_clock::now();
            ernie::validate_text_down_weights(argv[1]);
            std::cout << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << '\n';
            return 0;
        }
        require(argc==1,"Unexpected arguments");
        fs::create_directory(temp);
        std::ifstream file(ERNIE_TEXT_DOWN_FIXTURE);
        const std::string original((std::istreambuf_iterator<char>(file)),{});
        require(!original.empty(),"Missing graph fixture");
        const auto derived=ernie::vector_text_down_graph(original,64);
        std::ifstream large_file(fs::path(ERNIE_TEXT_DOWN_FIXTURE).parent_path()/"text-down-s2048.param");
        const std::string large((std::istreambuf_iterator<char>(large_file)),{});
        require(ernie::vector_text_down_graph(large,2048).find("ErnieTextDown")!=std::string::npos,"2048 template rejected");
        require(derived.find("ErnieTextDown gemm_6 1 1 72 73")!=std::string::npos,"Missing replacement");
        rejects([&]{ernie::vector_text_down_graph(original,32);});
        rejects([&]{ernie::vector_text_down_graph(original,2048);});
        rejects([&]{ernie::vector_text_down_graph(original+"Input extra 0 1 extra\n",64);});
        for (const auto& pair: {std::pair<std::string,std::string>{"10=-1","10=0"},{"0=4096 1=64","0=4096 1=63"},{"5=1","5=0"}})
        {
            auto bad=original;auto index=bad.find(pair.first);require(index!=std::string::npos,"Missing mutation target");
            bad.replace(index,pair.first.size(),pair.second);rejects([&]{ernie::vector_text_down_graph(bad,64);});
        }
        for (bool bf16: {false,true})
        {
            const auto p=temp/"weights";structural_weights(p,bf16);ernie::validate_text_down_weights(p.string());
            { std::ofstream append(p,std::ios::binary|std::ios::app);append.put(0); }
            rejects([&]{ernie::validate_text_down_weights(p.string());});
            fs::resize_file(p,3072*4+3);rejects([&]{ernie::validate_text_down_weights(p.string());});
            structural_weights(p,bf16);
            { std::fstream bad(p,std::ios::binary|std::ios::in|std::ios::out);bad.seekp(3072*4);bad.put(1); }
            rejects([&]{ernie::validate_text_down_weights(p.string());});
        }
        for (const auto* params: {"0=3071 1=0 2=28311552","0=3072 1=1 2=28311552",
                                 "0=3072 1=0 2=28311551","0=3072 1=0.0 2=28311552","0=3072 1=0 2=28311552 9=1"})
        { ncnn::Net n;n.opt=options();require(!ernie::register_text_down(n),"Register failed");require(n.load_param_mem(graph(params).c_str())!=0,"Bad parameters accepted"); }
        ncnn::Net net;net.opt=options();require(!ernie::register_text_down(net),"Register failed");
        require(!net.load_param_mem(graph(good).c_str()),"Load graph failed");
        auto* layer=net.layers().back();
        ncnn::Mat wrong(10);ncnn::ModelBinFromMatArray wrong_bin(&wrong);
        require(layer->load_model(wrong_bin)!=0,"Wrong weight shape accepted");
        ncnn::Mat weights(28311552);weights.fill(0.f);
        for (int n=0;n<3072;++n) weights[n*9216+4000+n]=1.f;
        weights[0]=std::numeric_limits<float>::quiet_NaN();ncnn::ModelBinFromMatArray nonfinite(&weights);
        require(layer->load_model(nonfinite)!=0,"Nonfinite weights accepted");weights[0]=0.f;
        ncnn::ModelBinFromMatArray bin(&weights);require(!layer->load_model(bin),"Load weights failed");
        require(!layer->create_pipeline(net.opt),"Create vector pipeline failed");
        ncnn::Mat input(9216,3);
        for (int r=0;r<3;++r) for (int k=0;k<9216;++k) input.row(r)[k]=float((k+17*r)%251-125)/128.f;
        ncnn::Mat result;require(!layer->forward(input,result,net.opt),"Forward failed");
        require(result.dims==2 && result.w==3072 && result.h==3 && result.elempack==1,"Output layout differs");
        for (int r=0;r<3;++r) for (int n=0;n<3072;++n)
            require(result.row(r)[n]==input.row(r)[4000+n],"Matrix/row exact oracle differs");
        ncnn::Mat bad(9215,3);require(layer->forward(bad,result,net.opt)!=0,"Wrong input width accepted");
        input[0]=std::numeric_limits<float>::infinity();require(layer->forward(input,result,net.opt)!=0,"Nonfinite input accepted");
        auto low=net.opt;low.use_bf16_storage=true;require(layer->forward(input,result,low)!=0,"BF16 accepted");
        fs::remove_all(temp);std::cout<<"text_down_contract passed\n";return 0;
    }
    catch(const std::exception& e) { fs::remove_all(temp);std::cerr<<e.what()<<'\n';return 1; }
}

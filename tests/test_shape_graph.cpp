// SPDX-License-Identifier: MIT
#include "shape_graph.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
static void require(bool ok) { if(!ok) throw std::runtime_error("Shape graph assertion failed"); }
template<class F> static void rejected(F f) { bool threw=false;try {f();} catch(const std::exception &) {threw=true;} require(threw); }
int main(int argc,char **argv)
{
    using namespace ernie;
    try {
        if(argc==6 && (std::string(argv[1])=="--instantiate" || std::string(argv[1])=="--instantiate-runtime")) {
            std::ifstream file(argv[3],std::ios::binary);
            if(!file) throw std::runtime_error("Missing template");
            std::string graph((std::istreambuf_iterator<char>(file)),{});
            std::cout<<instantiate_shape_graph(argv[2],graph,model_config(argv[4]),model_config(argv[5],std::string(argv[1])=="--instantiate-runtime"));return 0;
        }
        if(argc!=1) throw std::runtime_error("Use --instantiate kind graph source.cfg target.cfg");
        require(shape_graph_sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        require(shape_graph_sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        rejected([]{shape_graph_sha256(std::string(1024*1024+1,'x'));});
        ModelConfig a{4,4,32,272},b{64,64,64,64};
        require(reviewed_shape_config(a) && reviewed_shape_config(b));
        require(reviewed_shape_config(ModelConfig{86,48,64,64}));
        require(!reviewed_shape_config(ModelConfig{48,86,64,64}));
        require(reviewed_shape_config(ModelConfig{64,64,32,64}));
        validate_runtime_model_config(ModelConfig{128,64,2048,2048});
        rejected([]{validate_model_config(ModelConfig{128,64,2048,2048});});
        rejected([]{validate_runtime_model_config(ModelConfig{128,128,32,64});});
        rejected([&]{instantiate_shape_graph("text","",a,b);});
        rejected([&]{instantiate_shape_graph("unknown","",a,a);});
        rejected([&]{instantiate_shape_graph("vae","7767517\n1 1\n",a,a);});
        rejected([&]{instantiate_shape_graph("vae","",a,ModelConfig{32,32,32,32});});
        rejected([]{validate_model_config(ModelConfig{128,128,32,32});});
        rejected([]{validate_model_config(ModelConfig{-1,4,32,32});});
        std::cout<<"Reviewed graph contract checks passed; no model inference\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}

// SPDX-License-Identifier: MIT
#include "model_package.h"
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
int main(int argc,char**argv){
    try {
        if(argc==4 || argc==5){
            ernie::ModelPackage package(argv[1],std::stoi(argv[2]),std::stoi(argv[3]));
            const auto cfg=package.config();
            const auto source_cfg=package.source_config();
            std::optional<ernie::ModelPackage> reference;
            if(argc==5) reference.emplace(argv[4],std::stoi(argv[2]),std::stoi(argv[3]));
            const auto component=[&](const std::string &name,const std::string &kind) {
                auto actual=package.component(name,kind);
                if(reference) {
                    const auto expected=reference->component(name,kind);
                    std::istringstream a(actual.param_text),b(expected.param_text);
                    std::string av,bv;
                    while(a>>av) {
                        if(!(b>>bv) || av!=bv) throw std::runtime_error("Instantiated graph differs: "+name);
                    }
                    if(b>>bv) throw std::runtime_error("Instantiated graph is incomplete: "+name);
                }
                return actual;
            };
            int graphs=0;
            for(const auto &kind:{std::string("text"),std::string("dit")}){
                const int count=kind=="text"?25:36;
                const auto stem=kind=="text"?"text":"block";
                for(int i=0;i<count;++i){char index[3];std::snprintf(index,sizeof(index),"%02d",i);
                    auto c=component(kind+"/block-"+index+"/"+stem+".ncnn.param",kind);
                    if(c.param_text.empty()||!std::filesystem::is_regular_file(c.weight_path))throw std::runtime_error("Missing component");++graphs;
                }
            }
            for(const auto &kind:{std::string("input"),std::string("output")}){component("dit/"+kind+"/head.ncnn.param",kind);++graphs;}
            component("vae/head.ncnn.param","vae");++graphs;
            auto moved=std::move(package);
            if(moved.config().packed_width!=cfg.packed_width || moved.source_config().packed_width!=source_cfg.packed_width)
                throw std::runtime_error("Move lost runtime/source configuration");
            if((cfg.packed_width!=source_cfg.packed_width || cfg.packed_height!=source_cfg.packed_height) &&
                (moved.has_file("vae/encoder.ncnn.param") || moved.has_file("vae/encoder.ncnn.bin")))
                throw std::runtime_error("Wrong-size encoder exposed for runtime target");
            if(moved.file("tokenizer/tokenizer.json").empty())throw std::runtime_error("Missing tokenizer binding");
            bool rejected=false;try{moved.file("../escape");}catch(const std::exception&){rejected=true;}
            if(!rejected)throw std::runtime_error("Unsafe name accepted");
            std::cout<<"schema="<<moved.schema()<<" width="<<cfg.packed_width*16<<" height="<<cfg.packed_height*16
                     <<" source_width="<<source_cfg.packed_width*16<<" source_height="<<source_cfg.packed_height*16
                     <<" graph_texts="<<graphs<<" reference_graphs_equal="<<bool(reference)<<"; no model inference\n";
            return 0;
        }
        bool rejected=false;try{ernie::ModelPackage p("/__ernie_missing_package__");}catch(const std::exception&){rejected=true;}
        if(!rejected)throw std::runtime_error("Missing package accepted");
        std::cout<<"model package error contract passed\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

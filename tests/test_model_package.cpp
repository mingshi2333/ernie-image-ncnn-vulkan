// SPDX-License-Identifier: MIT
#include "model_package.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>
int main(int argc,char**argv){
    try {
        if(argc==4){
            ernie::ModelPackage package(argv[1],std::stoi(argv[2]),std::stoi(argv[3]));
            const auto cfg=package.config();
            int graphs=0;
            for(const auto &kind:{std::string("text"),std::string("dit")}){
                const int count=kind=="text"?25:36;
                const auto stem=kind=="text"?"text":"block";
                for(int i=0;i<count;++i){char index[3];std::snprintf(index,sizeof(index),"%02d",i);
                    auto c=package.component(kind+"/block-"+index+"/"+stem+".ncnn.param",kind);
                    if(c.param_text.empty()||!std::filesystem::is_regular_file(c.weight_path))throw std::runtime_error("Missing component");++graphs;
                }
            }
            for(const auto &kind:{std::string("input"),std::string("output")}){package.component("dit/"+kind+"/head.ncnn.param",kind);++graphs;}
            package.component("vae/head.ncnn.param","vae");++graphs;
            auto moved=std::move(package);
            if(moved.file("tokenizer/tokenizer.json").empty())throw std::runtime_error("Missing tokenizer binding");
            bool rejected=false;try{moved.file("../escape");}catch(const std::exception&){rejected=true;}
            if(!rejected)throw std::runtime_error("Unsafe name accepted");
            std::cout<<"schema="<<moved.schema()<<" width="<<cfg.packed_width*16<<" height="<<cfg.packed_height*16<<" graph_texts="<<graphs<<"; no model inference\n";
            return 0;
        }
        bool rejected=false;try{ernie::ModelPackage p("/__ernie_missing_package__");}catch(const std::exception&){rejected=true;}
        if(!rejected)throw std::runtime_error("Missing package accepted");
        std::cout<<"model package error contract passed\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}

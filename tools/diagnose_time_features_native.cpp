// SPDX-License-Identifier: MIT
// CPU-only observation of the production schedule and feature functions.
#include "denoiser.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
uint32_t bits(float x) { uint32_t value; std::memcpy(&value,&x,4); return value; }
int main(int argc,char **argv) {
    try {
        if(argc!=2 && argc!=3) throw std::invalid_argument("Require new output filename [zero-based step]");
        if(std::ifstream(argv[1]).good()) throw std::invalid_argument("Output exists");
        size_t consumed=0;
        const int step=argc==3 ? std::stoi(argv[2],&consumed) : 6;
        if(step<0 || step>=8 || (argc==3 && consumed!=std::strlen(argv[2])))
            throw std::invalid_argument("Step must be an integer in [0,7]");
        const auto schedule=ernie::FlowSchedule::turbo(8);
        const float value=schedule.timesteps.at(step);
        const auto decimal=std::to_string(value);
        const float reparsed=std::stof(decimal);
        const auto features=ernie::timestep_features(value);
        std::ofstream out(argv[1],std::ios::binary);
        out.write(static_cast<const char*>(features.data),4096*4);out.close();
        if(!out)throw std::runtime_error("Output write failed");
        std::cout<<std::setprecision(17)<<"{\"step\":"<<step<<",\"steps\":8,\"timestep\":"<<value
                 <<",\"timestep_bits\":"<<bits(value)<<",\"std_to_string\":\""<<decimal
                 <<"\",\"reparsed_bits\":"<<bits(reparsed)<<"}\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}

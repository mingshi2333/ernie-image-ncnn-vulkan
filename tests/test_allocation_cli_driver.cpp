// Exercises the actual CLI control flow with bounded fake inference/I/O.
// No model, real Vulkan instance, or numerical kernel is used here.
#define main allocation_cli_main
#include "../cli/main.cpp"
#undef main
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#ifdef ERNIE_CLI_ALLOCATION_METRICS
static std::uint64_t instance=0;
namespace ncnn { std::uint64_t get_gpu_instance(){return instance;} }
struct FakeAllocation {
    std::string mode;
    FakeAllocation(const std::string& m):mode(m) {
        if(mode=="no-allocation")return;
        instance=1;std::uint8_t uuid[16]{};
        if(mode!="missing-identity")ernie::allocation_device_identified(1,0,0x10de,123,1,2,"fake \"device\"\n",uuid);
        if(mode=="conflicting-identity")ernie::allocation_device_identified(1,0,1,2,3,4,"changed",uuid);
        ernie::allocation_allocator_created(1,2);
        ernie::allocation_allocator_role(1,2,ernie::AllocationRole::Weight);
        ernie::allocation_memory_created(1,2,3,35,{1,0,1,false,1});
    }
    ~FakeAllocation(){if(mode=="no-allocation" || mode=="live-allocation")return;ernie::allocation_memory_destroyed(1,2,3);ernie::allocation_allocator_destroyed(1,2);instance=0;}
};
#endif
namespace ernie {
DiagnosticInfo diagnose(const std::string&){return {};}
void verify_model(const std::string&){}
void verify_pe_model(const std::string&){}
GenerationResult generate(const GenerationRequest& r,const ProgressCallback&) {
#ifdef ERNIE_CLI_ALLOCATION_METRICS
    FakeAllocation allocation(r.model);
#endif
    if(r.model.find("report-failure")!=std::string::npos) {
        const char* p=std::getenv("ERNIE_TEST_REPORT_PATH");if(!p)throw std::runtime_error("missing test report path");
        std::ofstream f(p);f << "preserve-existing-report";f.close();
    }
    if(r.model.find("generation")!=std::string::npos)throw std::runtime_error("original generation failure");
    GenerationResult out;out.image={1,1,{2,4,6}};return out;
}
}
namespace ernie::cli {
RgbImage read_image(const std::filesystem::path&,const std::array<uint8_t,3>&){throw std::runtime_error("original input failure");}
RgbImage resize_image(const RgbImage& x,int,int,const std::string&,const std::array<uint8_t,3>&){return x;}
void write_image(const std::filesystem::path& p,const RgbImage& x) {
    if(p.filename().string().find("image-failure")!=std::string::npos)throw std::runtime_error("original image failure");
    std::ofstream f(p,std::ios::binary);f.write((const char*)x.pixels.data(),x.pixels.size());f.close();
}
}
int main(int argc,char** argv){
#ifdef ERNIE_CLI_ALLOCATION_METRICS
 for(int i=1;i<argc;++i)if(std::string(argv[i])=="borrowed-instance")instance=1;
#endif
 return allocation_cli_main(argc,argv);
}

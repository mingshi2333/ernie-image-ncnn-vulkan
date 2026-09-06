// SPDX-License-Identifier: MIT
#include "shape_graph.h"
#include <array>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
extern "C" int ernie_shape_sha256(const unsigned char *, std::size_t, unsigned char *);
namespace ernie
{
std::string shape_graph_sha256(const std::string &bytes)
{
    std::array<unsigned char,32> digest{};
    if (ernie_shape_sha256(reinterpret_cast<const unsigned char *>(bytes.data()),bytes.size(),digest.data()))
        throw std::invalid_argument("Graph text exceeds hash limit");
    static constexpr char hex[]="0123456789abcdef";
    std::string out;
    for(auto c:digest) { out+=hex[c>>4];out+=hex[c&15]; }
    return out;
}
using Fields=std::map<std::string,std::string>;
using Rules=std::map<std::string,Fields>;
static const std::map<std::string,Rules> rules={
    {"text", {{"gemm_0", {{"7","text_bucket"}}}, {"gemm_1", {{"7","text_bucket"}}}, {"gemm_2", {{"7","text_bucket"}}}, {"gemm_3", {{"7","text_bucket"}}}, {"gemm_4", {{"7","text_bucket"}}}, {"gemm_5", {{"7","text_bucket"}}}, {"gemm_6", {{"7","text_bucket"}}}, {"reshape_10", {{"2","text_bucket"}}}, {"reshape_11", {{"2","text_bucket"}}}, {"reshape_12", {{"2","text_bucket"}}}, {"reshape_13", {{"1","text_bucket"}}}, {"unsqueeze_18", {{"1","text_bucket"}}}, {"unsqueeze_19", {{"1","text_bucket"}}}}},
    {"dit", {{"gemm_0", {{"7","total_tokens"}}}, {"gemm_1", {{"7","total_tokens"}}}, {"gemm_2", {{"7","total_tokens"}}}, {"gemm_3", {{"7","total_tokens"}}}, {"gemm_4", {{"7","total_tokens"}}}, {"gemm_5", {{"7","total_tokens"}}}, {"gemm_6", {{"7","total_tokens"}}}, {"reshape_12", {{"2","total_tokens"}}}, {"reshape_13", {{"2","total_tokens"}}}, {"reshape_14", {{"2","total_tokens"}}}, {"reshape_15", {{"1","total_tokens"}}}, {"unsqueeze_20", {{"1","total_tokens"}}}, {"unsqueeze_21", {{"1","total_tokens"}}}}},
    {"input", {{"reshape_7", {{"0","image_tokens"}}}, {"gemm_0", {{"7","dit_text_tokens"}}}}},
    {"output", {{"gemm_0", {{"7","total_tokens"}}}, {"slice_0", {{"-23310","image_tokens_array"}}}, {"reshape_5", {{"0","packed_width"}, {"1","packed_height"}}}}},
    {"vae", {{"reshape_99", {{"0","vae_pixels"}}}, {"reshape_100", {{"0","vae_width"}, {"1","vae_height"}}}}},
};
static const Fields hashes={{"text","dda8db13b6e20a00c1133485c3be2ef37db56b1db058e42a405b51d24fc64b0e"},{"dit","51db4065837c28e235784fd0dd196f109ca86822143c025a44ee9372240e4132"},{"input","3710a502802138e8605ca54a9132fb9a9a1888841586c1d7067f7043769bbecb"},{"output","98bf1afc154dd05cf419aafad4aa383af311e44658e644d0c322ffb50431533c"},{"vae","6d68a0f10423b6ca243e229c8a9e8fe98c442b0d061ad5d38667f752cd9eb48f"}};
static Fields dimensions(const ModelConfig &c)
{
    validate_model_config(c);
    if(!reviewed_shape_config(c)) throw std::invalid_argument("Unreviewed static shape; independent evidence pending");
    int image=c.packed_width*c.packed_height;
    return {{"packed_width",std::to_string(c.packed_width)},{"packed_height",std::to_string(c.packed_height)},
            {"text_bucket",std::to_string(c.text_bucket)},{"dit_text_tokens",std::to_string(c.dit_text_tokens)},
            {"image_tokens",std::to_string(image)},{"total_tokens",std::to_string(image+c.dit_text_tokens)},
            {"image_tokens_array","1,"+std::to_string(image)},{"vae_width",std::to_string(c.packed_width*2)},
            {"vae_height",std::to_string(c.packed_height*2)},{"vae_pixels",std::to_string(image*4)}};
}
std::string instantiate_shape_graph(const std::string &kind,const std::string &graph,
                                    const ModelConfig &source,const ModelConfig &target)
{
    if(graph.size()>1024*1024) throw std::invalid_argument("Graph text too large");
    auto source_values=dimensions(source),target_values=dimensions(target);
    if(!rules.count(kind)) throw std::invalid_argument("Unknown graph kind");
    if(kind=="text" && source.text_bucket!=target.text_bucket)
        throw std::invalid_argument("Text bucket needs its independent exported template");
    std::istringstream input(graph);std::string line,canonical,result;
    std::set<std::string> nodes;std::set<std::pair<std::string,std::string>> seen;
    while(std::getline(input,line))
    {
        std::istringstream words(line);std::vector<std::string> f;std::string word;
        while(words>>word) f.push_back(word);
        auto changed=f;
        if(f.size()>1)
        {
            if(!nodes.insert(f[1]).second) throw std::invalid_argument("Duplicate graph node");
            auto rule=rules.at(kind).find(f[1]);
            if(rule!=rules.at(kind).end())
            {
                if(f.size()<6) throw std::invalid_argument("Malformed graph node");
                std::size_t start=0;
                try { auto a=std::stoll(f[2]),b=std::stoll(f[3]);
                    if(a<0 || b<0 || a>10000 || b>10000) throw std::invalid_argument("Invalid arity");
                    start=4+static_cast<std::size_t>(a+b);
                } catch(const std::exception &) { throw std::invalid_argument("Invalid graph arity"); }
                if(start>f.size()) throw std::invalid_argument("Missing graph connections");
                for(const auto &field:rule->second)
                {
                    std::size_t index=0,count=0;
                    for(std::size_t i=start;i<f.size();++i)
                        if(f[i].substr(0,f[i].find('='))==field.first) { index=i;++count; }
                    if(count!=1 || f[index]!=field.first+"="+source_values.at(field.second))
                        throw std::invalid_argument("Unreviewed shape parameter");
                    std::string upper=field.second;
                    for(char &c:upper) if(c>='a' && c<='z') c=char(c-'a'+'A');
                    f[index]=field.first+"="+upper;
                    changed[index]=field.first+"="+target_values.at(field.second);
                    seen.insert({rule->first,field.first});
                }
            }
        }
        auto append=[](std::string &out,const std::vector<std::string> &parts) {
            for(std::size_t i=0;i<parts.size();++i) { if(i) out+=' ';out+=parts[i]; } out+='\n';
        };
        append(canonical,f);append(result,changed);
    }
    std::size_t expected=0;for(const auto &rule:rules.at(kind)) expected+=rule.second.size();
    if(seen.size()!=expected || shape_graph_sha256(canonical)!=hashes.at(kind))
        throw std::invalid_argument("Unknown complete graph hash");
    return result;
}
} // namespace ernie

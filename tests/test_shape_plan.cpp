// SPDX-License-Identifier: MIT
#include "shape_plan.h"
#include <array>
#include <limits>
#include <stdexcept>
#include <iostream>
static void require(bool ok) { if (!ok) throw std::runtime_error("Shape assertion failed"); }
template<class F> static void rejected(F f) { bool threw=false; try { f(); } catch(const std::exception &) { threw=true; } require(threw); }
int main()
{
    using namespace ernie;
    ShapeContract c;
    auto s=ShapePlan::create(c,1376,768,1080);
    require(s.width==1376 && s.height==768 && s.packed_width==86 && s.packed_height==48);
    require(s.latent_width==172 && s.latent_height==96 && s.image_tokens==4128);
    require(s.text_bucket==2048 && s.valid_text_tokens==1080 && s.total_tokens==6176 && s.valid_tokens==5208);
    require(s.packed_latent_bytes==4128*128*4 && s.rope_elements==6176*128 && s.mask_elements==6176*6176);
    for (int t : {1,32,33,64,65,2048}) require(ShapePlan::create(c,16,16,t).text_bucket==(t<=32?32:t<=64?64:2048));
    for(auto wh : {std::array<int,2>{16,2048},{2048,16},{2048,1024},{1024,2048}}) require(ShapePlan::create(c,wh[0],wh[1],1).width==wh[0]);
    for(auto wh : {std::array<int,2>{0,16},{15,16},{17,16},{2049,16},{2048,2048},{-16,16}}) rejected([&]{ShapePlan::create(c,wh[0],wh[1],1);});
    for(int t : {0,-1,2049}) rejected([&]{ShapePlan::create(c,16,16,t);});
    rejected([&]{ShapePlan::create(c,std::numeric_limits<std::int64_t>::max(),16,1);});
    c.text_buckets={32,128,2048};rejected([&]{ShapePlan::create(c,16,16,33);});
    for (const auto &buckets : {std::vector<int>{}, {64,32}, {32,32}, {128}})
        rejected([&]{ShapePlan::create({buckets},16,16,1);});
    const auto padded = ShapePlan::create({{32},64},64,64,15);
    require(padded.text_bucket==32 && padded.dit_text_tokens==64 && padded.total_tokens==80 && padded.valid_tokens==31);
    rejected([]{ShapePlan::create({{32},64},64,64,33);});
    require(ShapePlan::create({{64,2048}},512,384,15).text_bucket==64);
    require(ShapePlan::create({},2048,1024,2048).total_tokens==10240);
    require(ShapePlan::create({},2048,1024,2048).host_weights);
    require(!ShapePlan::create({},1024,1024,2048).host_weights);
    require(ShapePlan::create({},1376,768,1080).host_weights);
    auto maximum=std::numeric_limits<std::size_t>::max();
    rejected([&]{checked_shape_product(maximum,2);});rejected([&]{checked_shape_sum(maximum,1);});
    require(checked_shape_product(maximum,0)==0 && checked_shape_product(maximum,1)==maximum && checked_shape_sum(maximum,0)==maximum);
    std::cout << "Checked shape, bucket selection and memory policy passed; no inference in this test\n";
}

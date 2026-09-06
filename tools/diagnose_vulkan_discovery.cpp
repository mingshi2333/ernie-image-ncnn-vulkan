// SPDX-License-Identifier: MIT
// No model construction, loading, dispatch, tensors or shader operations.
#include "gpu.h"
#include <chrono>
#include <iostream>
#include <thread>
static void hold(){std::this_thread::sleep_for(std::chrono::seconds(2));}
int main(){
 std::cout<<"BEFORE_CREATE"<<std::endl;hold();
 int status=ncnn::create_gpu_instance();
 if(status){std::cerr<<"CREATE_FAILED "<<status<<std::endl;return 1;}
 const int index=ncnn::get_default_gpu_index();
 auto device=ncnn::get_gpu_device(index);
 if(!device){ncnn::destroy_gpu_instance();return 2;}
 std::cout<<"DEVICE_READY "<<index<<std::endl;hold();
 ncnn::destroy_gpu_instance();std::cout<<"DESTROYED"<<std::endl;hold();
 return 0;
}

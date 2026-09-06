# O1 第一切片：独立记账组件与分配覆盖方案

本片完成基础组件，**不是O1全部完成**，不接block_sequence/dit/denoiser/vae/pipeline，不改变ncnn，不构建build-dev，不跑模型或GPU。使用既有隔离worktree，按P2计划的executing-plans步骤执行本次明确授权的有界切片。

## 已实现接口与口径

`src/execution_metrics.{h,cpp}`为标准C++17独立库，不依赖ncnn。所有注册、记账和snapshot由同一mutex保护。`register_allocator(device, allocator, role, domain, scope)`返回device/allocator/generation身份；活跃同身份重复注册拒绝，注销必须无live allocation，再使用同device/allocator地址得到新generation，旧token拒绝。每笔以allocator身份+handle定位，重复登记、未知/重复释放、零handle/size及整数溢出拒绝。正确free后原生handle重新使用是新分配事件；观测钩子仍必须按实际alloc/free顺序交付，不应异步重排事件。

`AllocationDomain`明确区分LogicalRequest、AllocatorBlock、VulkanMemory；三者各自live/peak，不能相加称物理峰值。未观察域是nullopt，不是0；注册且没有分配才是0。scope必填且保留角色/域/allocator身份、当前live分配及其bytes/可选alignment、历史allocator累计与peak。alignment缺失为unavailable；提供时必须正2次幂，仅作元数据，不按其推算或四舍五入实际bytes。真正VulkanMemory bytes必须来自VkMemoryAllocateInfo.allocationSize，调用方不能传Mat请求替代。跨allocator峰值按事件时间的总live累加，不是把各allocator峰值相加。注册只表示观察到该scope，不能自行宣称覆盖完整进程。内存类型/heap/host-import分类需要后续钩子元数据，本片没有伪造GPU现场值。

`record_interval(event,phase,start,finish,gpu_duration)`接受verify/read/prepare/read_prepare/upload/compute/wait/download；统一event ID与IO计数跨方法去重，时间倒序拒绝。host是interval之和，可能重叠，**不是端到端wall**。GPU只接真正时间查询结果；缺失保持nullopt，一组样本只要有一项缺失，GPU聚合也保持unavailable，保留samples/gpu_samples说明部分覆盖。read_prepare明确保留ncnn无法拆分的合并口径，不同时再填read+prepare。`record_io`明确计提交/上传/下载，未观察为nullopt，显式全0观察才是0。本片没有自动映射旧stats或生成JSON CLI。

Collector生命周期应限单次请求/实验，事件ID去重集合与已注册allocator历史保留至销毁；不提供会静默遗失仍存活分配的reset。它不是长期无限增长的进程遥测服务。

## 实际测试

独立小compiler单进程构建，C++17/O2/-Wall/-Wextra/-Werror/pthread；原工程CMake精确增添独立library和execution_metrics_cpu test target，无运行库接线。因为build-dev由root占用，本次CTest使用独立`outputs/execution-metrics-o1-v1`指向真实已编译测试二进制；`ctest --test-dir outputs/execution-metrics-o1-v1 -R execution_metrics --output-on-failure` **1/1通过**。source、编译命令/版本、binary SHA与源码snapshot保存在identity.json。未执行主build-dev的完整CMake集成测试，后续空闲时再做。

测试：10+20-free10+15得到live=peak=35；重复/未知释放、重复alloc、活跃allocator注销拒绝；同handle不同device独立；allocator地址复用时旧generation失效；真实`new[]`64字节池仅分配一次，两个10字节逻辑lease复用，lease释放后物理池live仍64，直到真正delete/free才归零，Vulkan域保持unavailable。该host池测试是实际内存分配/复用，不是实际Vulkan allocator测试。计时覆盖GPU缺失/混合、重叠host区间、计数0/未知、溢出不破坏既存计数；两个线程各1000次事件且并发snapshot，最终2000提交/4000上传字节、无live泄漏。线程数最多2。

## pinned ncnn 最小窄统计钩子提案（未实现）

候选pin `6a1bf000f363714839a36793addc8c879d3d899e`；读取原主目录third_party/ncnn。allocator.cpp SHA `601d69dab40823366fa6aa2be00c8e37bb0f1e960fe96323c36daed887c4fda2`，net.cpp SHA `258de463be0a3ad82072eea2fce714f4ebd3ed914f147732f05b9865e74ff57e`。参考pin `f6f734f44d66f469fefee9ee401fd1cb5e3d573e`独立source，allocator.cpp SHA `fe8f6bc2fcee095ed172763445ddcd6c8e39070a561eaf5048fdb433246dd6b0`，net.cpp SHA `4e6e1aa5dc13cb0f7ec24df9b3327ed022233414b6ced0675549550d66cece9a`。

1. 最窄且可靠的位置是`VkAllocator::allocate_memory`(候选allocator.cpp:446)、`allocate_dedicated_memory`(:466附近)、`allocate_import_host_memory`(:491)的vkAllocateMemory成功返回之后。捕获device、allocator(this)+generation、VkDeviceMemory handle、allocationSize、memoryTypeIndex、heapIndex、flags、imported_host、角色；失败不能登记成功分配。pAllocator/VkAllocationCallbacks只统计driver的host分配，不是这里需要的VkDeviceMemory大小，不能替代。
2. 统一包装allocator.cpp所有真实vkFreeMemory调用，在执行实际free时移除对应handle。包括Blob clear/free池释放、Weight clear专属/普通池、Staging/WeightStaging回收；fastFree把子块放回budget池时**不**减少VulkanMemory live。实际VkDeviceMemory可服务多个Mat/VkBuffer子区域，不能按Mat析构或VkBuffer释放重复free计数。
3. 角色入口：Net::load_model候选net.cpp:2053/2057创建`VkWeightAllocator`和`VkWeightStagingAllocator`；Extractor默认取得blob/staging于net.cpp:2927/2936。VulkanDevice初始化和acquire路径gpu.cpp:4177/4178、4850/4887也必须覆盖，缓存通常使用调用方独立blob allocator，应由创建者标Cache；不能仅给pipeline传入一个自定义blob allocator就声称覆盖weights/staging。跟踪器需在VulkanDevice创建之前激活，且在其allocator清理之后结束，否则预热池/延迟释放会形成未观察live。共享长期device需明确baseline live，不可把未结束请求的释放计入新请求。
4. 钩子函数示意`on_memory_allocate(device,allocator_generation,memory,size,type,heap,flags,imported_host,role)`和`on_memory_free(device,allocator_generation,memory)`；必须noexcept，不让统计改变模型行为。身份错误/队列溢出标记统计invalid，不能吞错后仍报告完整峰值。当前ExecutionMetrics接口会抛异常，ncnn桥接必须捕获并将整个测量标invalid；不能让异常越过GPU库边界。
5. 对于Linux两侧相同的上述函数与free点可应用相同语义、小型独立diff，不更改算子/精度/权重/同步。参考low-vram的import-host VkDeviceMemory必须保留heap/flags/import类别，不能把它全部称设备本地VRAM，也不能称它无主机开销。总VkDeviceMemory、device-local heap、host-visible/imported分别报告；同一allocation不能按多个用途重复求总和。CPU RSS仍由外部进程计数，与Vulkan allocation非同一维度。
6. Android HardwareBuffer路径还有独立vkAllocateMemory(:2384)及失败/释放分支；本轮限定Linux，若不覆盖该路径须明确平台排除，不能宣称跨平台全覆盖。gpu.cpp中VkDummyAllocator等初始化资源需单列Other或明确作为共同排除的设备固定成本。ncnn以外driver/桌面其他进程不在该口径；整卡采样可补充预算观察，不能替代该进程的峰值。

后续必须对同一saved输入分别执行插桩开/关，验证byte-identical输出，并核对净live与allocator清理，双方同范围后才有M结论。详细插桩仅内存/诊断轮，正式速度轮关闭；GPU时间需完成后的timestamp query及设备timestampPeriod/valid bits，不把host enqueue/wait当GPU执行。单块/64×64逐位开关验证、真实VkDeviceMemory覆盖、成本阶段接线、指标CLI、三种真实负载、双方测量均pending。

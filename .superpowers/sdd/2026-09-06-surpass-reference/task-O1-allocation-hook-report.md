# O1 第二切片：可选实际Vulkan分配观察适配

本片在48a72a5基础上实现可选observer、两pin派生源码工具和独立CMake入口。**未运行真实GPU分配/模型，也未证明开关插桩后的输出逐位一致；O1仍未完成。**没有编辑原始third_party、参考ncnn或兄弟项目；两原checkout最终git status均clean。没有使用build-dev构建/配置，仅为独立单TU语法检查读取既有生成头。

## 可用接口

`AllocationMeasurementSession`提供单个活动测量session；`snapshot()`返回available/valid、可选总VkDeviceMemory live/peak、按(device,memoryTypeIndex,heapIndex,propertyFlags,imported_host,heapFlags)分组统计，以及上一片完整allocator角色/身份统计。compiled OFF或者尚未观察任何allocator时，实际分配数值为unavailable，不能变成看似完整的0。观察到allocator但未分配时允许已观察scope为0；**available不表示整个进程/驱动覆盖完成**。权重、blob、staging自动由派生构造器标识；用户持有的专用session cache可在任何分配前调用`allocation_allocator_role(...,Cache)`，已有live时改role会使测量invalid。

全局活动session指针受mutex保护，callback与session销毁串行，使用上一片generation身份处理allocator地址复用。allocation成功事件捕获实际VkMemoryAllocateInfo allocationSize及内存属性；free事件位于原vkFreeMemory之后。pool fastFree仅缓存子块时无事件，所以保留块仍计入live。host-import保持单独flag/heap类别，绝不把所有VkDeviceMemory称为设备本地VRAM。driver CPU malloc、ncnn以外分配/桌面进程不在覆盖范围，外部RSS和整卡预算采样仍是独立口径。

所有observer函数noexcept：未知owner、duplicate memory、错误free/生命周期、计数异常被捕获并使session.valid=false；内部互斥异常通过独立原子标志使快照invalid。观测失效后不再发布新的貌似可靠计数。实际Vulkan调用/返回/释放无条件保持，observer不拥有、不释放任何模型内存。异常快照可能保留部分数值，但valid=false，消费者必须拒绝用它计算M。Session必须先于待观察device/allocator创建，并在实际清理结束后取最终快照；中途开启遇到未知历史allocation会invalid，而不是偷偷从0开始。结束session不会等待GPU或替调用方销毁allocator；调用方负责原本的同步/生命周期。

## 派生源码与默认OFF

新增`cmake/ErnieAllocationMetrics.cmake`，选项`ERNIE_ENABLE_ALLOCATION_METRICS`默认OFF；OFF立即返回原source path，不查找Python、不产生副本、不构建/链接observer库、不更改ncnn编译标准。`cmake/Dependencies.cmake`仅最后add_subdirectory位置经过prepare函数，随后条件attach。ON采用C++17 observer和独立ernie-execution-metrics库，ncnn仅为包含接口启用C++17；正式逐位回归仍pending。ON限定Linux Vulkan，Android和NCNN_PYTHON显式拒绝。

`cmake/derive_allocation_ncnn.py`核验root git pin、子模块pin、每个tracked working file与git blob对应，完整路径/SHA256规范JSON聚合与固定digest一致后才复制。候选pin6a1bf000...含7793条记录，full SHA `e5e8d449ddb09e2183faca8be4ab535899ff8bc463665fea209c9b05604ef4d5`；参考pin f6f734f4...含7292条，full SHA `a3340d10902b5102ee19fe1af2d1c317947b111595e7a92884ccf5236a0013af`。计数包含gitlink身份；glslang的初始化源码递归认证/复制，未启用Python的pybind11只固定gitlink、不复制。原文件不写入，输出必须在原source之外。只有allocator.cpp被修改，其余逐文件内容不变。重新配置必须验证derived内容、完整文件集合和patch记录，patcher变化要求新build目录，不覆盖旧证据。

两pin使用同一狭窄变换：3个成功allocation helper返回前上报（一般、dedicated、host-import），10个真实vkFreeMemory调用之后上报（Linux实际适用8个，另2个Android分支因平台明确禁用），base构造/析构注册/注销，4类derived构造标role。不修改VkResult/内存handle，不取消任何原调用，不调整数值shader、精度、模型或运行同步。两份allocator.cpp完整SHA白名单、原free调用数和每函数唯一return核验均fail closed。ncnn构造器接受空device时hook不会新增空指针解引用，只把无效统计身份标invalid。

独立参考项目可复用同一cmake prepare/attach函数，以其原ncnn路径生成自身pin的派生副本；当前没有改参考主CMake或替换baseline二进制。两pin的派生patch与全清单provenance都保留，后续需要原参考/插桩参考相同输入输出验证，才能称测量范围已建立。

## 已执行测试与证据

- fake backend ON/OFF均验证原allocation返回值不变、失败返回0不上报、原free次数不变；统计duplicate/未知free/带live销毁变invalid后，后端后续allocate/free仍执行；host-import/type/heap分组、35bytes live/peak、allocator复用、2线程2000次事件正确。
- 两份pin的Python源码认证/patch合同各3/3通过，核验原vkAllocateMemory/vkFreeMemory调用数不变，所有free事件位于真实free之后，未知源码拒绝。
- 独立CMake configure/build使用最多2jobs，ON/OFF `ctest --test-dir outputs/allocation-hooks-o1-v3/build --output-on-failure` **2/2通过**；OFF脚本对不存在source路径原样返回，证明不额外解析/复制源码。
- 两份最终派生allocator.cpp分别使用各自已有generated platform头进行`c++ -std=c++17 -fsyntax-only`成功，无模型运行。该检查不是完整ncnn链接/运行验证。
- 早期首次patch误设free数16而实际10，在复制前明确拒绝，修正为真实逐项计数；首次peer语法检查误选generated include路径报platform.h缺失，改用peer build/ncnn/src后通过。初版小编译-Werror发现单行if缩进歧义，已修复；保留旧输出目录，不覆盖。

最终证据`outputs/allocation-hooks-o1-v3/{identity.json,ctest.log,configure.log,build.log,ncnn-derived/allocation-metrics.patch}`及`outputs/allocation-hooks-peer-o1-v2/ncnn-derived/`。下一步需队列空闲后完整插桩ncnn构建、真实allocator池复用/释放验证、同一单块/64×64开关逐位输出比较，再由root连接--metrics-json。尚无任何实际GPU live/peak数值、无性能收益或M结论。

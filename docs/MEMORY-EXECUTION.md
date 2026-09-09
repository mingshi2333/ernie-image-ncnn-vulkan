# ERNIE 的权重预取、RAM 缓冲区与分配失败恢复

本文说明 Vulkan DiT 执行阶段的内存组织，供其他 ncnn 项目复用。
实现分成三个明确的职责：准备下一块权重、为新缓冲区选择内存、从已完成的去噪步骤恢复。
它们共用可用内存查询和错误分类，生命周期仍由一次生成请求管理。

本文依据当前开发代码及本机 Linux 实际测试撰写。完整模型对照与三平台 CI 的最新状态见[验证记录](../artifacts/2026-09-08/memory-execution/README.md)；实现存在不等于所有平台、模型和内存压力条件都已验证。

## 1. 先确认参数的作用范围

这些控制项用于 `--device vulkan` 的 DiT；文本编码、CPU 提示增强 PE 和 VAE 保持各自的执行路径。
默认精度仍为 FP16。使用可选权重缓存或预取时，需要明确选择 FP32。
MiB 表示 1,048,576 字节，CLI 与 C++ API 使用相同单位和默认值。

| CLI 参数 | `GenerationRequest` 字段 | 默认值 | 作用 |
|---|---|---:|---|
| `--dit-weights auto\|device\|host` | `dit_weights` | `auto` | DiT 头和块的权重位置请求 |
| `--gpu-reserve-mib N` | `gpu_reserve_mib` | 512 | 权重估计和缓冲区准入使用的显存预留 |
| `--dit-cache-mib N` | `dit_cache_mib` | 0 | 跨去噪步骤保留部分已准备的 RAM 权重；0 关闭 |
| `--dit-prefetch-mib N` | `dit_prefetch_mib` | 0 | 提前准备至多下一块权重；0 关闭 |
| `--gpu-memory auto\|device\|host` | `gpu_memory` | `auto` | DiT 激活与工作区缓冲区的位置策略 |
| `--gpu-spill-mib N` | `gpu_spill_mib` | 2048 | 实际分配的 host-visible Vulkan 缓冲区字节上限 |
| `--ram-reserve-mib N` | `ram_reserve_mib` | 3072 | 缓存、预取和 host 缓冲区准入共同使用的 RAM 预留 |
| `--oom-retries N` | `oom_retries` | 3 | 分配失败后的最大重试次数，允许 0..3 |

缓存和预取要求 Vulkan FP32，权重模式必须为 `auto` 或 `host`。
显式 `device` 保留用户选择；显式激活 `host` 要求非零 spill 预算。
在 `auto` 下把 spill 预算设为 0，可以禁止新增 host 缓冲区；`--oom-retries 0` 禁止失败后的重试。
这些配置的合法性由公共生成入口检查，直接调用 C++ API 也适用。

下面是启用一块权重预取的示例，模型包需要支持所选尺寸；输出文件必须是新路径：

```sh
build/ernie-image --model /path/to/model-package \
  --prompt "a red apple on a wooden table" \
  --width 512 --height 512 --device vulkan --precision fp32 \
  --dit-weights auto --dit-prefetch-mib 1024 \
  --gpu-memory auto --gpu-spill-mib 2048 --oom-retries 3 \
  --report-json outputs/prefetch-report.json --output outputs/prefetch.png
```

1024 MiB 是示例预取预算，实际能否准入取决于该块权重大小、准备后计账和当时 RAM 余量。
调用方也可设置 `request.dit_prefetch_mib`、`request.gpu_memory` 等字段，再调用 `ernie::generate(request)`。
公共接口见 [pipeline.h](../include/ernie/pipeline.h)，详细运行方式见 [RUNNING.md](RUNNING.md)。

## 2. 从同步流式执行增加一块异步准备

原执行顺序是“加载块 i → 计算块 i → 释放 → 加载块 i+1”。
[block_sequence.cpp](../src/block_sequence.cpp) 保留该顺序的所有权边界，只增加一个 `std::future<PreparedBlock>`。
主线程拿到当前 Net 后，才允许后台准备下一块；不会启动第二个并行预取任务。

具体流程如下：

1. 若下一块已在 `WeightSession` 的空闲缓存中，直接等待之后命中，避免重复准备。
2. 用 `2 × 权重文件字节数 + 64 MiB` 估计加载和准备开销，检查独立预取预算及 RAM 预留。
3. 主线程作出 host 权重请求，再把组件文件、不可变选项和设备传给后台任务。
4. 后台 Net 自己拥有权重分配器和局部统计；不修改缓存、主线程统计或放置策略回调。
5. 消费下一块时取得 future，检查实际权重驻留和计账，再重新检查 RAM 余量。
6. 可准入的 Net 进入正常执行/缓存租约；无法确认、实际回退到 device-local 或超预算的结果会被丢弃，回到同步加载。

实际检查由 [weight_session.cpp](../src/weight_session.cpp) 中的 `dit_host_weight_inspector()` 完成。
它只接受已识别的 FP32 图，检查 Vulkan 权重的 memory type，对共享底层分配去重计数，并加上 CPU 权重和 64 MiB 开销余量。
这个余量是明确的计账规则，不能当成实测的 Net、驱动或进程 RSS 峰值。

估计在加载前使用，实际检查在加载后使用。因此 `--dit-prefetch-mib` 约束准入，无法严格约束文件读取、临时打包和驱动内部准备过程的瞬时占用。
预取、缓存和激活 spill 各有预算，RAM 预留通过实时查询共享；三项预算相加也不等于进程内存上限。
文件页缓存、线程栈、ncnn 临时数据和系统其他进程仍会占用内存。

后台读取或图加载失败会在取得 future 时传播。当前计算或用户回调提前退出时，future 在作用域退出前等待后台结束，避免设备和分配器先被销毁。
进入分配失败恢复后，当前尝试的预取和缓存均被释放，后续尝试关闭二者。

## 3. 重叠的是准备工作，提交仍受同步保护

ncnn 的权重准备包含文件读取、解包、图与流水线准备，以及可能需要的 Vulkan 上传。
下一块的 CPU 准备可以与当前块计算重叠，但共享队列必须遵守 Vulkan 的外部同步要求。

[ncnn_vulkan_failure.h](../cmake/ncnn_vulkan_failure.h) 中的 `QueueLease` 为 `VkCompute` 和 `VkTransfer` 提供同一个递归互斥锁。
取得队列、提交和等待在这个保护范围内串行执行，结束后归还队列；单队列设备也按此处理。
因此不能把“后台预取”解释成两个 GPU 队列必然同时计算/传输，也不能据此承诺吞吐提升。

`prefetch_overlap_seconds` 记录准备区间与主线程当前块执行区间的交集。
它是主机时钟观察到的重叠时间，不是 GPU profiler 得到的并行时长，也不是节省的总时间。
加载与计算统计存在重叠时，不应把各项简单相加作为端到端耗时。

已有负面结果需要保留：缓存策略修正后命中从 13 增至 42，耗时却从 378.970 秒变为 384.364 秒；文件缓存的内存限额压力仍存在。
这不是受控配对测速，既不能据此断言策略导致变慢，也不能宣称命中增加就会加速。
原始范围和数据见 [后续测量与负面结果](../artifacts/2026-09-07/cache-file-lru/README.md)。
新预取需要单独比较同一二进制、同一输入、预取关闭/开启的时间、RSS、文件页压力和输出一致性。

## 4. 激活和工作区如何使用 RAM

[vulkan_memory.cpp](../src/vulkan_memory.cpp) 的 `AdaptiveVkAllocator` 同时作为 DiT 的 blob 与 workspace 分配器。
模型仍使用原来的 Vulkan shader；host 模式改变缓冲区的内存类型，计算继续由 GPU 完成。

`auto` 在新分配时检查实际 compute heap 的 budget 减 usage，再扣除 GPU 预留。
能够满足时使用 ncnn 的 device 池；不足或实际 device 分配发生内存错误时尝试 host 缓冲区。
显存预算查询不可用时允许正常 device 分配尝试；RAM 余量不可用时拒绝 host 准入。
`device` 禁止 host 回退，`host` 直接请求 host-visible 缓冲区。

host 路径创建带 `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST` 用途的 Vulkan buffer，查出真实 `VkMemoryRequirements` 后再检查预算。
随后选择 `HOST_VISIBLE | HOST_COHERENT` memory type，优先避开 `DEVICE_LOCAL`，完成分配、绑定和映射。
计账使用实际 Vulkan allocation size，包含驱动要求的对齐；它可能大于张量字节数。
每个 Vulkan 调用失败都必须在继续使用 buffer、mapped pointer 或输出张量前传播。

判断显存余量时，不能拿已经放到 RAM 的张量所属 heap 当成 GPU compute heap。
`compute_memory_budget_reader()` 通过一个小 storage buffer 的兼容 memory type 查询定位 compute heap，不分配 backing memory。
RAM 余量则来自 [host_memory.cpp](../src/host_memory.cpp) 的独立系统查询。

这个分配器支持混合位置，而 ncnn 的 `mappable` 属性属于整个 allocator。
因此它保留 staging 上传/下载路径，避免某个 host buffer 可映射就错误推断所有 device buffer 都可映射。
Vulkan image 分配仍委托给 device 池；当前生产 DiT 使用 buffer 路径。

## 5. 已释放的张量为何还暂时占用 RAM

CPU 上最后一个 `VkMat` 引用消失时，已经记录的命令仍可能引用它的 buffer。
如果 `fastFree()` 当场调用 `vkFreeMemory()`，后续提交就会读取已经释放的内存。

host buffer 的 `fastFree()` 只标记为 retired，计账也继续保留。
只有相关命令完成，并且命令对象已 reset 或销毁后，才调用 `reclaim_completed()` 真正释放。
头、块以及 FP32 注意力内部查询分块的同步边界负责回收，活动张量始终保留。
失败路径先同步队列/设备，再让本次尝试的张量、缓存和分配器结束生命周期。

这套机制在分配时决定新缓冲区的位置；已有 device buffer 不会在任意时刻被透明换页到 RAM。
需要整体改变位置时，恢复器释放失败尝试，再从 CPU 检查点重建所需缓冲区。
它没有实现任意活动 buffer 的在线迁移、OS swap 管理或磁盘缓存。

## 6. 以完整去噪步骤为恢复边界

[vulkan_denoise.cpp](../src/vulkan_denoise.cpp) 的 `denoise_with_recovery()` 管理一次尝试的分配器、流水线缓存、权重缓存和 GPU 张量。
每次 Euler 更新完成且 latent 有限后，将 FP32 master latent 下载到 CPU；下载及检查成功才提交检查点。
因此即使未启用 trace，Vulkan 去噪也有每步检查点传输成本。

分配失败后的处理顺序是：

1. 保留最近完整步骤的 CPU latent 和下一步的绝对索引，未完成步骤不提交。
2. 确保失败尝试的已提交工作结束，释放其预取、缓存、临时缓冲区和执行对象。
3. `auto` 权重在恢复中请求 host；`auto` 激活在非零 spill 预算下改用 host，显式 device 选择保持不变。
4. 关闭可选缓存与预取，将 FP32 非 Flash 注意力查询行数从 128 逐次降到 64、32、16。
5. 重新上传检查点与条件，从失败步骤继续，沿用原来的完整时间表。

最多是首次执行加三次重试。设置为 0 时首次内存错误直接结束；到达上限后返回明确失败。
重试不会改变分辨率、生成步数、seed、模型权重、精度或每行访问的完整 K/V。
查询分块只限制同时计算的 Q 行数；FP16/BF16 的原生 Flash 路径和 KV cache 路径没有套用这项缩块策略。
缩块保持相同数学定义；实际数值一致性仍需要同输入回归验证。

恢复点位于当前进程的内存中。进程被操作系统杀死、关闭或重启后，不具备持久化检查点恢复能力。
内存总量仍不足时，有限次数的 RAM 回退和分块缩减也可能全部失败。

## 7. 先保留错误语义，才能安全恢复

ncnn 的通用 `-1` 可能表示图、参数、文件或执行失败，不能直接解释成 Vulkan OOM。
应用层分别检查两个错误域：ncnn 的 `-100` 视作分配失败；原始 Vulkan 的 host/device OOM 才进入内存恢复。
`VK_ERROR_DEVICE_LOST`、非内存错误、非有限结果和用户观察器异常直接结束请求。
观察器即使抛出 `std::bad_alloc`，也不会被当成应重跑模型的信号。

上游若先把原始 Vulkan 错误压成 `-1`，应用层已无法可靠区分。
[ErnieNcnnFailures.cmake](../cmake/ErnieNcnnFailures.cmake) 因此在构建目录生成经过源码散列认证的 allocator、command、net 编译单元，保留精确分配失败语义，并检查空上传目标、staging 映射及提交失败。
实际 Vulkan OOM 通过 `std::bad_alloc` 传播，设备丢失保持致命错误；不会广泛把所有错误改为可重试。
替换发生在 ncnn target 内，安装的 SDK 使用相同实现；原 submodule 文件不被改写。

此前两个权重测试退出码均为 0，却共出现 10 条 Vulkan validation 错误。
其中 `VUID-vkBindBufferMemory-memory-02985` 来自 host 导入内存与 buffer 外部句柄声明不匹配，`VUID-vkCmdCopyBuffer-srcBuffer-00118` 来自权重 buffer 缺少 transfer-source 用途。
[ErnieNcnnAllocator.cmake](../cmake/ErnieNcnnAllocator.cmake) 对固定来源修正这两个契约，并处理 Windows 的不同 host 导入路径。
这些属于 Vulkan 使用错误，不能解释为用户 RAM 太小。

全量校验还发现旧 BF16 SDPA shader 使用了设备未声明支持的 BF16 accumulator 协作矩阵类型，即使数值测试通过也属于无效 Vulkan 使用。
[ErnieNcnnBf16Sdpa.cmake](../cmake/ErnieNcnnBf16Sdpa.cmake) 认证 C++ 和 shader 来源，只禁用 BF16 SDPA 的 cooperative 选择，继续使用原生 BF16 Flash。
它没有修改注意力数学或 FP32/FP16 的选择，但暂时失去该 BF16 分支的协作矩阵加速；修正前的 BF16 完整模型数据不能直接作为当前路径的证据。

源码认证先规范化 CRLF，再检查完整散列；ncnn 升级后不匹配会停止配置，要求重新审查。
[ErnieVulkanValidation.cmake](../cmake/ErnieVulkanValidation.cmake) 同时检查 stdout/stderr 中的 Validation Error 和 VUID，防止再次出现“退出码成功但 Vulkan 契约已失败”。
开启 `ERNIE_TEST_VULKAN_VALIDATION` 会请求 Khronos validation layer；没有可用驱动或不支持的精度仍按测试原有规则跳过。
由于 CTest 的跳过码会优先于输出失败正则，Linux Vulkan CI 还独立扫描完整 LastTest.log，防止带 validation 错误的跳过被漏记。

## 8. 平台差异、统计与验证范围

Linux RAM 余量同时考虑 `MemAvailable` 与所有有限 cgroup-v2 上级限额，干净文件页只给予保守回收额度。
Windows 同时受可用物理内存、提交额度和地址空间限制；Windows Job 或 Wine 场景目前返回余量不可确认。
当前其他平台，包括 macOS，尚无生产 RAM 余量读取器，因而拒绝 host 缓冲区准入；显式 `host` 也不能绕过余量检查。
平台微型测试可注入可信的余量读取器来验证 Vulkan buffer 行为，这不代表 macOS 或 Windows Job/Wine 的生产自动 RAM 恢复已经可用。

UMA 设备可能只有同时带 `HOST_VISIBLE` 和 `DEVICE_LOCAL` 的内存类型。
host buffer 此时不代表获得了额外的独立 RAM 池；报告分别记录 host device-local 与 host non-device-local 次数。
权重预取/缓存的实际驻留检查更严格，会拒绝 device-local 权重；驱动具备 Vulkan 能力不代表满足这些准入条件。

`--report-json` 的 `weight_prefetch` 提供 started/used/skipped、计账峰值和观察到的重叠时间。
`gpu_memory` 提供 buffer 分配次数、实际 host 峰值、回退和失败计数；`memory_recovery` 提供重试数与最终查询行数。
这些统计不覆盖整个 GPU、完整进程 RSS 或文件缓存峰值。比较性能时应保留端到端耗时、实际内存监控和输出身份各自的范围。

[test_memory_execution.cpp](../tests/test_memory_execution.cpp) 使用真实微型 Vulkan 图验证预取与恢复：三块 2×2 Gemm，以及 128 通道、1×1 latent 的四步 Euler 模型。
测试包含实际步骤后注入失败、检查点逐位对比、非零起始步骤、次数上限、后台错误、取消、缓存交互及回调异常。
[test_vulkan_memory.cpp](../tests/test_vulkan_memory.cpp) 负责 host buffer 计算、预算拒绝和延迟释放等分配器契约。
受控预算拒绝和故障注入用于检验行为，不等于真实整卡耗尽或系统 OOM-kill 实验。

本次 Linux Vulkan 完整 CTest 61/61 通过、0 跳过，Khronos validation 无错误；纯 CPU 的 37/37 测试同样通过。
其中预取、RAM buffer 与检查点恢复使用实际 Vulkan 运算，FP32 注意力的四种分块大小经过独立 FP64 参考检查。
初次测试图缺少 Split 的崩溃、旧 host buffer 的 10 条校验错误和旧 BF16 accumulator 的 20 条校验错误均保留在验证记录里，没有覆盖成成功结果。
同输入完整模型回归正在运行，三平台 CI 待本次推送；此处不把已有旧版本结果记为本次完成。
性能结论还需要预先定义的重复对照，不能从新增机制或单次成功推导。

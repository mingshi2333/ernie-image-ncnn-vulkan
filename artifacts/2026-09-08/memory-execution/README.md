# DiT 预取、RAM 缓冲区与检查点恢复

2026-09-08 开始实现，09-09 继续验证。代码基线 `00b6e64`，ncnn 仍固定为 `3b7bdba7`；使用构建目录中的已认证兼容修正，不改写 submodule。

三项实现及后续修正已推送至获授权的私有验证分支。正常与混合内存 FP32 已完成整图：各 25/25 张量和 PNG 与原基线逐位相同，混合配置实际使用 288 个非 device-local host buffer、280 次预取全部被消费。[v2 完整结果](v2/README.md)保留实际内存与耗时；它证明配置可用，不构成加速保证。

BF16 完整校验暴露了另一处 Gemm cooperative 类型不匹配。修正后的源码 `7296bfe` 已通过本机 **62/62 CTest，0 跳过、0 Vulkan 校验错误**，包括新 Gemm 独立 FP64/BF16 舍入参考测试；详情见 [BF16 Gemm 前后验证](bf16-gemm/README.md)。修正前 `5b57e9d` 的[三平台 CI](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34369154353)已全部通过；最终 `7296bfe` 的 [CI 34373124004](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34373124004)五个作业也全部成功，225 通过、35 能力跳过、0 失败，另有 115 项 HTTP/清单测试通过；实际 checkout、SDK 与原始产物身份已[独立核对](v3/final-ci/summary.json)。单独冻结的 [v3 BF16 完整模型](v3/README.md)已结束，零校验错误，但仍仅 17/25 张量通过、PNG MAE 1.294207255/max 143，继续保持实验状态。两个版本分别记录，不用旧结果代替新结果。

[默认 FP16 完整回归](v4/README.md)也已结束：未指定精度的命令实际使用 FP16，25 份张量及 PNG 与旧 FP16 逐位一致，官方仍为 23/25、PNG MAE 0.235983531/max 109。全部模型与 CI 任务均已结束。四例分别冻结在两个源码版本，src/include/cli 的 87 个文件逐位相同，差别限于 BF16 构建守卫及新增测试。不能把三套计划说成同一个二进制。

## 实现与默认值

- 一次最多后台准备下一块 FP32 权重，独立 Net/分配器和预算，消费前再次核对 RAM 余量和真实权重驻留。预取默认 0 MiB，缓存仍默认关闭。CPU 准备可以重叠，GPU 队列提交受同步保护。
- DiT 的新激活与工作区 buffer 默认 `auto`，GPU 预算不足或分配失败后允许使用 GPU 可访问的 RAM；host buffer 默认上限 2048 MiB。真实分配大小包括对齐和待安全回收的 buffer。
- 每步保存完整 FP32 CPU latent，最多三次重试；失败后重建该次执行、关闭缓存/预取，自动模式改用 host，并把 FP32 非 Flash 查询分块依次缩为 64/32/16。完整 K/V、分辨率、步数、精度和时间表不变。
- 通用 ncnn `-1`、设备丢失、非有限输出、图/文件错误和用户回调异常直接失败；只有明确分配错误恢复。ncnn 分配/上传/提交路径必须保留错误类型、检查空缓冲区，并在销毁前完成必要同步。

作用域是 Vulkan DiT buffer。已有活动 VkMat 不会任意在线换页；image、文本、PE、VAE 和进程重启后的恢复不包含在内。macOS 尚无生产 RAM 余量读取器，Windows Job/Wine 同样拒绝无法确认的 host 准入；微型测试注入读取器不代表这些平台的完整模型恢复已验证。详见 [技术教程](../../../docs/MEMORY-EXECUTION.md)。

## 已完成的验证

| 范围 | 当前结果 | 记录 |
|---|---|---|
| 本机 Linux Vulkan / RTX 4060 Laptop | 最新 62/62 通过，0 跳过，Khronos validation 无错误；首版 61 项日志也保留 | `bf16-gemm/final-ctest.xml`、`bf16-gemm/final-last-test.log` |
| 本机 Linux 纯 CPU | 37/37 通过，0 跳过 | `cpu-ctest.xml` |
| 实际微型 Vulkan 图 | host buffer 运算、预算拒绝、延迟回收；异步预取、缓存/取消/异常交互；第三步失败后从最近完成步骤恢复，输出逐位相同 | 上述 CTest 的 `vulkan_memory_buffers`、`memory_execution_vulkan` |
| Vulkan 故障传播 | 分配/绑定/映射/上传/提交故障注入与共享队列使用；非 OOM 保持失败 | `ncnn_failure_vulkan`、`allocator-runtime.json` |
| FP32 查询分块 | 4 种输入 × 128/64/32/16 行，原 FP64 与未切分基线门槛不变 | `attention_workspace_vulkan` |
| BF16 SDPA 源码修正 | 8 项来源、CRLF、拒绝脏输入、target 唯一性、CPU 绕过与实际编译检查通过 | `bf16-guards.json` |
| BF16 Gemm 修正 | 13 项来源/编译检查，4 种矩阵形状 × cooperative ON/OFF 精确参考；新路径无 VUID | `bf16-gemm/source-guards.json`、`bf16-gemm/final-ctest.xml` |
| 完整 512 正常与混合 FP32 | 各 25/25 张量与 PNG 基线逐位相同，官方 25/25、PNG max 1 | `v2/full-comparison.json` |
| 完整 512 默认 FP16 | 25/25 张量与 PNG 同旧 FP16，官方仍 23/25、PNG max 109 | `v4/full-comparison.json` |
| 完整 512 新 BF16 | 运行有效、0 VUID，但官方 17/25、PNG max 143，数值门槛失败 | `v3/full-comparison.json` |
| 最终源码三平台 CI | 225 通过、35 能力跳过、0 失败；115 项 HTTP/清单通过 | `v3/final-ci/summary.json` |
| 四例独立回读 | 三份计划全部绑定、源码/派生单元/命令/资源/数值重新核对；执行有效，整体官方数值通过为 false | `execution-audit.json`、`audit_execution.py` |

[独立审计](execution-audit.json)重新读取 379 个唯一文件、约 23.97 GB 的来源与模型数据，重算四例各 4,981,760 个有限张量值及 PNG；与原比较逐项一致。它明确保留 v2 整批 `validation_failed` 及无效 BF16 的 5 条校验错误，不把选出的两个有效 FP32 结果改写为整批通过。

这些是受控预算和故障注入，未故意耗尽整卡或触发系统 OOM-kill。测试集还包含 API、CLI、JSON 报告和安装 SDK 消费检查；完整名称及每项结果以 XML 为准。

## 保留的失败与修正

1. 原始权重测试退出 0，但出现 10 条外部内存声明、transfer-source 用途错误。修正 buffer 契约，保留 `baseline-validation.json`；这与用户 RAM 大小无关。
2. 初次新 buffer 测试的图重复消费同一输入而缺少 Split，发生 SIGSEGV。修正测试图之后通过；原失败及调用栈保留，不把它解释成 OOM。
3. 全量 Vulkan validation 暴露两个旧缓存测试共 20 条 BF16 cooperative accumulator 能力错误。只关闭 BF16 SDPA 的该加速分支，保留原生 BF16 Flash；FP32、FP16 和其他层的 cooperative 选择不变。代价是这条 BF16 注意力路径暂时不使用协作矩阵加速。原数值通过并不允许忽略非法 Vulkan 用法，原日志 `initial-vulkan-errors.log` 保留；修正后 61 项全过。这只是当时的 SDPA 修正；v2 完整模型随后暴露 Gemm 的同类 VUID，v3 又以新路径独立验证，不能直接沿用历史数值。
4. 故障补丁与原有 allocation-metrics 派生 allocator 组合时，第一次编译失败；兼容两套源码结构后实际编译通过。分别保留 `metrics-initial-compile.log` 与 `metrics-fixed-compile.json`，不覆盖原失败。
5. CTest 的 skip 77 会优先于输出失败正则，所以 CI 额外扫描整个 LastTest.log；有 VUID 的跳过不能算干净通过。

## 首轮 v1 完整模型与全 RAM 部分结果

二进制 SHA256 `00f0e2112988f28d25480cb6bdaebd7594a0edf81c3b1d5d5a5fe9b07664b3b7`，357 个源码文件、152 项模型/参考/源码/脚本/二进制绑定在启动前验证。快照和完整输出位于 `/var/tmp/ernie-memory-execution-20260908-v1`。

同一 shared schema-3 模型、保存初始噪声、15-token 英文提示词、512×512、8 步、2 CPU 线程、原生 Vector 文本编码、Vulkan DiT 和 CPU direct VAE，trace 开启。三个预先确定的配置：

1. FP32 正常模式，host 权重、cache 0、prefetch 0、GPU reserve 512 MiB。
2. FP32 自动 RAM 模式，GPU reserve 8192 MiB、cache 1024 MiB、prefetch 1024 MiB，其他输入相同。8192 MiB 大于实际 8188 MiB 显存，目的是受控触发策略，未真实耗尽 GPU。
3. BF16 原生 Flash 回归，与先前 BF16 完整结果及相同官方参考比较。

保持既有 16 GiB cgroup、swap 0、主机可用 RAM 至少 3 GiB、整卡采样上限 6144 MiB、每次 1800 秒限制。FP32 比较先前认证的完整基线，BF16 比较先前 `candidate-bf16`，并独立重算官方 25 个张量和 PNG 的原门槛。

正常 FP32 已完成：25/25 张量和 PNG 均与历史基线逐位相同；官方比较仍为 25/25、PNG MAE 0.000361124674479、max1。全 RAM 配置完成前两步，已落盘的 10 个张量与历史基线全部逐位相同；没有完整 PNG，不能算完整通过。前两步分别为 231.491/228.332 秒，明显体现访问 RAM 的代价。主任务为修正已发现的池复用准入问题而主动停止本次执行；这不是 OOM 或数值失败，现场 cgroup max/OOM/OOM-kill 均为0。原 supervisor 按执行未完成记录失败。过程和停止原因见 `all-ram-interrupted/`。BF16 尚未执行。单次 trace 对照、未控制的文件页缓存和背景负载不能用来承诺提速。旧缓存命中改善但耗时未改善、cgroup 文件缓存压力的 [负面记录](../../2026-09-07/cache-file-lru/README.md) 继续保留。

源码快照的初版派生文件收集器只检查文件名，漏掉了编译目录中的四个副本。保留原空记录，`derived-compiled-units.json` 根据启动前冻结的 compile_commands 补齐其实际编译路径和完整摘要；没有重新构建或修改实验程序。

后续源码检查确认：实时显存预算不足时，原新分配检查会忽略 ncnn 池中可直接复用的空闲范围，从而转 RAM。已修正为只对需要新增 backing memory 的请求应用预算；实际测试验证同一 VkBuffer/VkDeviceMemory、连续空闲范围、活动区间、碎片合并与清空边界。本机 61/61 测试和严格校验通过；此问题与全 RAM 强制配置的访问延迟是两件事，该配置从未允许 device 分配。后续 v2 混合配置已证明实际 device/host 同时出现并完成整图；原极端配置仍保留为部分结果。

首轮 Linux Vulkan CI 的 40 条 VUID 只有一个类型：Ubuntu 校验层 1.3.275 不认识 `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES`，但 Mesa 25.2.8 已暴露该特性。61 项中 19 项因此失败、4 项按 BF16 能力跳过。工作流现在固定官方 LunarG Linux SDK 1.4.357.1，并核对下载 SHA256、实际校验层版本及 llvmpipe；不关闭校验，也不替换系统 loader/Mesa。隔离加载同一新版校验层后，本机 61/61、远程 Linux Vulkan 57 通过/4 BF16 跳过，均无校验错误。SDK 下载与静态检查、池复用及三平台原始记录见 `v2/`。

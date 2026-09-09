# DiT 预取、RAM 缓冲区与检查点恢复

2026-09-08 开始实现，09-09 继续验证。代码基线 `00b6e64`，ncnn 仍固定为 `3b7bdba7`；使用构建目录中的已认证兼容修正，不改写 submodule。

当前已完成实现和 Linux 小型测试。完整模型对照正在运行，三平台 CI 尚待本次推送；不能把这份准备记录当作最终出图或平台验收结果。

## 实现与默认值

- 一次最多后台准备下一块 FP32 权重，独立 Net/分配器和预算，消费前再次核对 RAM 余量和真实权重驻留。预取默认 0 MiB，缓存仍默认关闭。CPU 准备可以重叠，GPU 队列提交受同步保护。
- DiT 的新激活与工作区 buffer 默认 `auto`，GPU 预算不足或分配失败后允许使用 GPU 可访问的 RAM；host buffer 默认上限 2048 MiB。真实分配大小包括对齐和待安全回收的 buffer。
- 每步保存完整 FP32 CPU latent，最多三次重试；失败后重建该次执行、关闭缓存/预取，自动模式改用 host，并把 FP32 非 Flash 查询分块依次缩为 64/32/16。完整 K/V、分辨率、步数、精度和时间表不变。
- 通用 ncnn `-1`、设备丢失、非有限输出、图/文件错误和用户回调异常直接失败；只有明确分配错误恢复。ncnn 分配/上传/提交路径必须保留错误类型、检查空缓冲区，并在销毁前完成必要同步。

作用域是 Vulkan DiT buffer。已有活动 VkMat 不会任意在线换页；image、文本、PE、VAE 和进程重启后的恢复不包含在内。macOS 尚无生产 RAM 余量读取器，Windows Job/Wine 同样拒绝无法确认的 host 准入；微型测试注入读取器不代表这些平台的完整模型恢复已验证。详见 [技术教程](../../../docs/MEMORY-EXECUTION.md)。

## 已完成的验证

| 范围 | 当前结果 | 记录 |
|---|---|---|
| 本机 Linux Vulkan / RTX 4060 Laptop | 61/61 通过，0 跳过，Khronos validation 无错误 | `vulkan-ctest.xml`、`ctest-last.log` |
| 本机 Linux 纯 CPU | 37/37 通过，0 跳过 | `cpu-ctest.xml` |
| 实际微型 Vulkan 图 | host buffer 运算、预算拒绝、延迟回收；异步预取、缓存/取消/异常交互；第三步失败后从最近完成步骤恢复，输出逐位相同 | 上述 CTest 的 `vulkan_memory_buffers`、`memory_execution_vulkan` |
| Vulkan 故障传播 | 分配/绑定/映射/上传/提交故障注入与共享队列使用；非 OOM 保持失败 | `ncnn_failure_vulkan`、`allocator-runtime.json` |
| FP32 查询分块 | 4 种输入 × 128/64/32/16 行，原 FP64 与未切分基线门槛不变 | `attention_workspace_vulkan` |
| BF16 SDPA 源码修正 | 8 项来源、CRLF、拒绝脏输入、target 唯一性、CPU 绕过与实际编译检查通过 | `bf16-guards.json` |

这些是受控预算和故障注入，未故意耗尽整卡或触发系统 OOM-kill。测试集还包含 API、CLI、JSON 报告和安装 SDK 消费检查；完整名称及每项结果以 XML 为准。

## 保留的失败与修正

1. 原始权重测试退出 0，但出现 10 条外部内存声明、transfer-source 用途错误。修正 buffer 契约，保留 `baseline-validation.json`；这与用户 RAM 大小无关。
2. 初次新 buffer 测试的图重复消费同一输入而缺少 Split，发生 SIGSEGV。修正测试图之后通过；原失败及调用栈保留，不把它解释成 OOM。
3. 全量 Vulkan validation 暴露两个旧缓存测试共 20 条 BF16 cooperative accumulator 能力错误。只关闭 BF16 SDPA 的该加速分支，保留原生 BF16 Flash；FP32、FP16 和其他层的 cooperative 选择不变。代价是这条 BF16 注意力路径暂时不使用协作矩阵加速。原数值通过并不允许忽略非法 Vulkan 用法，原日志 `initial-vulkan-errors.log` 保留；修正后 61 项全过。完整 BF16 数值待本次模型对照，旧结果不能直接代替。
4. CTest 的 skip 77 会优先于输出失败正则，所以 CI 额外扫描整个 LastTest.log；有 VUID 的跳过不能算干净通过。

## 已冻结的完整模型对照

二进制 SHA256 `00f0e2112988f28d25480cb6bdaebd7594a0edf81c3b1d5d5a5fe9b07664b3b7`，357 个源码文件、152 项模型/参考/源码/脚本/二进制绑定在启动前验证。快照和完整输出位于 `/var/tmp/ernie-memory-execution-20260908-v1`。

同一 shared schema-3 模型、保存初始噪声、15-token 英文提示词、512×512、8 步、2 CPU 线程、原生 Vector 文本编码、Vulkan DiT 和 CPU direct VAE，trace 开启。三个预先确定的配置：

1. FP32 正常模式，host 权重、cache 0、prefetch 0、GPU reserve 512 MiB。
2. FP32 自动 RAM 模式，GPU reserve 8192 MiB、cache 1024 MiB、prefetch 1024 MiB，其他输入相同。8192 MiB 大于实际 8188 MiB 显存，目的是受控触发策略，未真实耗尽 GPU。
3. BF16 原生 Flash 回归，与先前 BF16 完整结果及相同官方参考比较。

保持既有 16 GiB cgroup、swap 0、主机可用 RAM 至少 3 GiB、整卡采样上限 6144 MiB、每次 1800 秒限制。FP32 比较先前认证的完整基线，BF16 比较先前 `candidate-bf16`，并独立重算官方 25 个张量和 PNG 的原门槛。

当前本节只有协议与启动前证据，结果后续归档。单次 trace 对照、未控制的文件页缓存和背景负载不能用来承诺提速。旧缓存命中改善但耗时未改善、cgroup 文件缓存压力的 [负面记录](../../2026-09-07/cache-file-lru/README.md) 继续保留。

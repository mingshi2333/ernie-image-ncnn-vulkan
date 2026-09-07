# 根据实时显存预算选择 DiT 权重位置

2026-09-07，新增的默认 auto 策略接入原生 DiT heads 和每个流式 block：每次加载前读取实际计算堆的 Vulkan 预算与本进程使用量，在估计余量不足时请求系统内存权重。完整 512×512 受控 RAM 回归完成，25 个 FP32 张量、4981760 个有限值与已验证原生基线逐字节一致，最终 PNG 也逐字节一致。本项没有改变官方数学、精度、步数、阈值或模型内容。

## 实现与边界

- `src/weight_placement.*` 独立管理预算读取、选择与请求计数，不持有权重、GPU 命令或后台线程。heads/blocks 在前一组件完成后选择新 Net 的权重位置，原有完成后释放流程保持。
- `--dit-weights auto|device|host` 默认 auto；host 表示 RAM，计算仍走 Vulkan。`--gpu-reserve-mib` 默认 512，是可调的额外余量，并非显存占用上限或模型硬件门槛。估计权重负载为已验证 BF16/FP32 文件大小的两倍；它不估计整个推理峰值。
- 剩余预算取 `max(heapBudget - heapUsage, 0)`。固定 ncnn 的 `get_heap_budget()` 只返回预算，未扣使用量，因此不直接使用它作为剩余显存。查询语义见 [Khronos 官方定义](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html)。数据是可变化的驱动估计，不保证后续分配成功。
- 保留 schema-3 大于 6144 **tokens** 的 RAM 偏好；这与测试 supervisor 的 6144 **MiB** 整卡保护线无关。驱动不支持预算查询时记录 unavailable，按原有尺寸偏好选择，手动模式仍可用。
- 公共 C++ 头仍只依赖标准库；新增请求字段放在末尾，保留旧位置聚合初始化。CLI/API 报告 placement requests，trace 增加 `weight-placement.txt`。固定 ncnn 的 host 分配仍可能内部退回 device，不能把请求计数冒充全覆盖的分配器测量。
- 尚未实现运行中激活向 RAM 迁移、已失败 Vulkan 命令的恢复、跨去噪步的有界准备权重缓存或自动 RAM 容量预算。大型激活/工作区仍需要显存。本项不关闭 O2、正式速度/内存指标或整体超过参考项目的目标。

## 小型验证

Vulkan 构建 5/5 受影响 CTests 通过：预算策略、实际 Vulkan 连续三个流式小型 GEMM（GPU→RAM→GPU）、公共 API、请求校验及 CLI。实际三个 GEMM 使用已知权重得到精确输出 `[10,-54]`；受控预算读取覆盖每块重新查询。测试还覆盖预算减用量、等值边界、用量超过预算、整数溢出、手动覆盖、查询缺失与文件元数据读取失败。真实驱动查询同时执行；模拟预算变化不代表实际耗尽显存。

CPU 构建 5/5（含安装后的外部 C++ 消费者）及 Vulkan 安装消费者 1/1 通过，共 11 个受影响 CTest 执行结果。最初多目标构建已完成生成器，但测试目标因原配置 `BUILD_TESTING=OFF` 不存在；启用测试、完成两种构建后上述检查全部通过。详见 `checks.json` 及三个测试日志。Windows/macOS 和其他精度的完整图像没有在本项重跑。

## 完整模型的受控 RAM 回归

使用同一共享 source32 包、15-token 原生 Vector 文本、64 个 DiT 文本槽、同一已保存初始 latent、8 步 Vulkan FP32 与 CPU direct VAE。测试冻结 352 个源码/二进制/输入/基线绑定，源目录隐藏、网络禁用、模型只读。候选编译时映射加载仍为 OFF。

为在不占满 GPU 的情况下覆盖自动分支，此次显式设置 `--dit-weights auto --gpu-reserve-mib 8192`。这是测试输入，**不把默认余量改成 8GiB，也不表示这台机器实际没有可用显存**。304 次（8 × [input head + 36 blocks + output head]）全部记录 `reason=budget`、请求 host、无查询缺失；记录的剩余预算范围为 5684789248..6341394432 字节，没有 ncnn host→device 分配回退日志。此结果不证明异步 OOM 恢复或长期压力下稳定性。

| 项目 | 结果 |
|---|---|
| 完整执行 | 退出 0，8 步及 VAE 完成 |
| 数值回归 | 25/25 张量，共 4981760 个元素全部有限、逐字节相同 |
| 最终图像 | 512×512，PNG 字节相同，最大通道差 0 |
| GPU 观测 | 整卡采样峰值 2363MiB，包含桌面/其他进程 |
| 进程组内存 | cgroup `memory.current` 采样峰值 4618027008B，含文件缓存，非进程 RSS |
| 主机余量 | 采样最低可用 14738100224B |
| 资源事件 | cgroup max=0、OOM=0、OOM-kill=0 |
| 监督区间耗时 | 526.741229893 秒；带 trace、CPU 2 核限额，不是正式速度测量 |

基线监督区间为此前 466.868 秒。本次没有表现出加速；缓存状态和观测时刻未控制，不能据此计算正式性能回退比例或显存改善比例。RAM 放置与减少反复读取/准备权重是不同改动，后者仍待实施。

候选二进制 SHA256：`48db9c4dfb9d24cd970d6732f0bfaa66a67caabd13faab1fb01f07e2790d194e`。
PNG SHA256：`bf87214d21461bfda96529830b28a5b7630ef0dcf307c0172f1666fc64044e28`。
冻结计划 SHA256：`4d811426f6ad0d112785df970dcf1dcece41af3707e9c0dd1694ffdbc94f85d5`。

本地大文件在 `/var/tmp/ernie-weight-placement512-v1`，原始逐时采样在其 `native/samples.jsonl`；权重、二进制、图像和大张量没有进入 Git。`inventory.json` 索引所选小文件及其散列。旧官方对照、长文本/中文/横图数值未过项和保护线终止记录均保留。本批所有模型进程已结束。

# 原生文本编码、去噪、VAE 与 PNG 验证

**完成：64×64 原生 prompt → PNG 的官方模块数值对照，以及本机一次 1024×1024 原生生成。** 固定 ERNIE-Image-Turbo、batch=1、8 steps、CFG=1、PE 关闭。1024 结果没有完整官方去噪对照，`quality_validated=false` 保持原值。

本目录由 `tools/collect_pipeline_evidence.py` 收集小型原始日志、fixture/模型 manifest、图结构和测量结果；权重、张量、运行器快照及图片保留在本机 `models/`、`outputs/`，不进入 Git。`SHA256SUMS` 封存目录内文件，`source-sha256.json` 对应收集时的源码，各次运行自己的 binary / validator / fixture 散列保留在原 JSON。**当前源码快照不代替历史运行版本**：64 数值运行后增加了 CLI 分项时间与输入 embeddings，1024 运行后又增加文件有限值提前检查和基准异常返回修正；这些没有重新标记旧结果的来源。

## 1024 实际出图

- 提示词：`A red apple on a wooden table, soft daylight, realistic photo.`
- 15 tokens（含 BOS），文本桶 32；4096 个图像 tokens + 64 个 DiT 文本位置。
- 原生 `std::mt19937` / `std::normal_distribution`，seed 42；实际初始 latent 与每步张量均保存。不能用相同整数 seed 推断与 PyTorch 噪声一致。
- CPU FP32 文本编码；Vulkan FP16 DiT，两个残差相加点和 skip 保持 FP32；FP32 Euler；CPU FP32 VAE，GroupNorm 归约 FP64。
- 本机图片：`outputs/pipeline1024-apple-fp16-v1/native.png`，1024×1024 RGB。
- PNG SHA-256：`06ab1162a81914ff25527719fba201e268b5a40b215c03aea469ba68e0698f9c`。
- 肉眼检查：木桌上有一只红苹果，窗边柔和日光，主体和场景符合该提示词。这只是单张视觉检查，没有建立感知质量分数或广泛提示词准确率。

运行记录：[request](runs/pipeline1024-apple-fp16-v1/request.json)、[result](runs/pipeline1024-apple-fp16-v1/result.json)、[原生日志](runs/pipeline1024-apple-fp16-v1/runner.log)、[资源](runs/pipeline1024-apple-fp16-v1/resources.log)。

| 项目 | 本次测量 |
|---|---:|
| 文本编码 | 12.8351 s |
| 第 1 步去噪 | 59.7523 s |
| 第 2–8 步 | 55.58–56.40 s / step |
| 8 步去噪合计 | 451.5973 s |
| VAE 和 PNG | 56.4258 s |
| 程序总耗时 | 522.174 s |
| 进程最大 RSS | 24,145,308 KiB = 23.0268 GiB |
| 整卡首个显存样本 | 1228 MiB |
| 整卡采样峰值 | 2605 MiB |
| 显存采样 | 100 ms，5203 个样本 |
| 系统已用 swap 增量 | 1,004,572 KiB = 0.9580 GiB |

GPU 为 RTX 4060 Laptop 8188 MiB，CPU 为 Ryzen 7 7745HX，主机 32GB RAM。完整环境见 [environment.json](environment.json)。运行包含 trace 保存，未做冷缓存控制或多次热运行统计。显存为 device 0 整卡总量，含其他程序，可能漏过短时峰值；系统 swap 也不能全部归因于此进程。不能称为精确本进程 VRAM、零 swap 或相对其他实现的提速结果。

## 64 完整数值门槛

参考使用固定版本官方 Mistral 文本模块、36 层 DiT、Euler、BN/unpack 和 VAE 分阶段执行，避免同时加载全部权重。模型/官方实现版本见 [锁文件快照](environment/sources.lock.json)。这里没有执行需要完整多模态/PE 权重的顶层官方 `from_pretrained` 一次调用，也未将自写 ncnn 输出作为参考。

原生程序读取同一份保存的初始 latent，独立编码相同 prompt；25 个中间/最终张量对照全部通过。原生 PNG 还与自身 FP32 decoded 张量的量化逐位核对。

| 检查 | 误差 | 判定 |
|---|---:|---|
| 初始 latent | 逐位一致 | 通过 |
| 文本特征 | NRMSE 6.2812e-6 | 通过 |
| 全部逐步预测与 Euler 输出 | 每项分别检查，最大预测 NRMSE 约 0.07678 | 通过 |
| 最终 latent | NRMSE 0.0524404 | 通过 |
| 解包后 latent | NRMSE 0.0522111 | 通过 |
| VAE decoded | NRMSE 0.0268799 | 通过 |
| PNG 像素 | MAE 1.46541/255，最大差 15/255 | 通过 |
| PNG 对原生 decoded 的量化 | 逐位一致 | 通过 |

FP16 门槛在执行前保存：NRMSE ≤ 0.15，最大绝对误差 ≤ `0.03 + 0.25 * max(abs(reference))`；PNG MAE ≤ 12、最大差 ≤ 80。文本与位置条件使用独立 `2e-4` 级门槛。该全张量尺度判据不是逐元素 allclose，也不意味着感知质量已评估。FP32、FP16 门槛均保留在 [gates.json](runs/pipeline64-apple-fp16-v3/gates.json)，实际结果见 [result.json](runs/pipeline64-apple-fp16-v3/result.json)。

64 图片是模糊纹理状数值 fixture，原生与官方均如此，不应称作有可用质量的苹果图。其 PNG SHA-256 为 `d9cd2369d09ae3fff4c8267ca37fb0538ef58fb9eb7254f8182c04e4fca08cc9`。参考 fixture 因最初原生失败而保存在 [旧运行的 reference](failures/pipeline64-apple-fp16-v1/reference/fixture.json)，后来通过全部散列检查后复用，未重写参考或放宽门槛。

## 组件证据

| 范围 | 结果与边界 |
|---|---|
| 时间特征 | 33 个有效 timestep 对照 + 5 个无效输入拒绝；最大绝对误差 3.0514e-5，最大 NRMSE 1.224e-6 |
| 36 blocks × 8 Euler，合成文本 | CPU FP32 最终 NRMSE 9.4732e-5，Vulkan FP32 2.6624e-5，FP16 0.0292165；三个配置通过 |
| 文本 hidden-state 选择 | 完整小模型 26 层、三个序列长度确认 `hidden_states[-2]` 是 block 24 输出，最后一个 hidden state 才是 final norm 输出 |
| 实际模型分派 | 官方 Mistral3 配置使用 `MistralModel` / `MistralAttention`；没有调用 Ministral3 的 llama4 scaling |
| 真实文本 block 0 | CPU FP32 / Vulkan FP32 / FP16 / BF16 四个配置通过 |
| 真实英文 prompt 的 25 层文本路径 | CPU NRMSE 6.2812e-6，Vulkan FP32 6.9130e-6，FP16 0.003128；三个配置通过 |
| 中文与空文本 | CPU 单独对照通过，分别 NRMSE 6.2892e-6、1.6017e-6 |
| 4160-token 前后处理 | 输入/输出层各四个配置通过；独立合成输入，不等于完整 1024 DiT 参考 |
| 64 VAE | 原始小图四个配置通过；这轮在后来的 FP64 CPU GroupNorm 归约修复之前 |
| 1024 VAE | 修复后 CPU NRMSE 2.0179e-6，最大绝对误差 1.2353e-5；独立官方目标分辨率参考通过 |

组件 JSON 保存在 `runs/` 下。此前 tokenizer 48 个样本、独立 DiT 141/144 配置和 pipeline cache 对照仍以 [旧组件报告](../components/README.md) 为准，不重新计作本轮执行。BF16 第 31、33、35 号独立块失败仍有效；本轮未完成 BF16 全去噪闭环。

## 保留的失败及修复

1. **真实文本下 FP16 残差溢出。** 首次 64 原生执行从第一步开始已出现 NaN，直到 PNG 的有限值检查才拒绝。其 [result](failures/pipeline64-apple-fp16-v1/result.json) 为失败。逐层保存真实输入定位到第 14 号 block（第 15 层），此前层全部有限，该层有一个无穷值。CPU FP32 同输入残差峰值 78784.765625，高于 FP16 上限 65504。保留 FP32 残差后峰值 78792、NRMSE 0.00028482，完整 64 闭环随后通过。详见 [逐层诊断](failures/real-activation-diagnostic-v1/result.json) 和 [定点复验](runs/residual-block14-fix-v1/result.json)。新增每步有限值检查，GPU 仅回传 128 个 float 状态，避免带坏 latent 继续运行。
2. **高分辨率 VAE 整图 pnnx 转换资源失败。** 官方 128×128 latent → 1024 参考及 wrapper 对照已完成，但 pnnx 形状执行持续增长主机内存和 swap，约 17 分钟后停止自己的转换进程。保留 [资源停止记录](logs/vae128-resource-stop.json)、[原日志](logs/export-vae-128x128-v1.log) 和 [转换 metadata](models/vae-128x128-v1/conversion.json)。后续方法是对完整图哈希核验后仅修改两处空间 reshape，并用目标官方参考验收；[特化记录](models/vae-128x128-specialized-v1/conversion.json) 明确不是整图 pnnx 导出成功。
3. **VAE 原生 CPU GroupNorm 大尺寸归约误差。** 原版本 NRMSE 2.5071e-5，高于既定 2e-5 门槛；关闭 Winograd 后 2.5122e-5，仍失败。改用 FP64 均值和中心方差归约后 2.0179e-6 通过。两次失败均保留；不将关闭 Winograd 宣称为修复。独立 VAE 的最大 RSS 24,614,984 KiB、耗时 52.50 s，只属于该组件运行。
4. **GPU 作业重叠导致分配失败。** 64 FP32 全流程和大尺寸 Vulkan VAE 被同时启动，发生 `vkAllocateMemory` 错误；完整流程返回 139 / SIGSEGV。执行安排已改为大 GPU 作业顺序运行。两份失败记录保留：[完整流程](failures/pipeline64-apple-fp32-v2/result.json)、[VAE](failures/vae-128x128-validation-v1/matrix.json)。该重叠运行不能证明单独 FP32 或 GPU VAE 超过设备容量；GNU time 的信号结果需结合进程退出码，不能只看其末尾 Exit status 字样。
5. **开发检查失败。** 第一次文本验证因 NumPy bool JSON 序列化失败，修正后重跑通过；第一次残差测试编译使用错误的 `load_model` overload，导致 CTest NotRun，修正后 17/17 通过。日志保留，不把未执行计为通过。

## 轻量检查与当前边界

- Vulkan 构建 [CTest 17/17](logs/ctest-residual-v2.log)，CPU 构建 [CTest 6/6](logs/ctest-cpu-generator-v1.log)。覆盖原生 KV cache、GELU、RMSNorm/LayerNorm、FP32 残差及 latent 有限值契约。
- [Python 契约 15/15](logs/contracts-pipeline-final-v1.log)：来源/shape/文件完整性、无损权重流、单文件和 Range 下载错误处理。
- [CLI 输入检查 12/12](runs/cli-input-contract-v1/result.json)：两种构建均拒绝 NaN/Inf embeddings、错误张量长度、Inf latent、超长 prompt 和无效步数；有效保存 embeddings 能进入 latent 检查而不加载文本权重。
- 当前源码完成 Python 语法检查；新增输入有限值检查后两种生成器重新构建，未重复执行未改变的重型数值矩阵。

32-token 桶、两个固定分辨率、CPU 默认 VAE、本机单次 1024 结果是当前边界。长文本、多提示词质量、高分辨率官方完整去噪对照、重复性能、精确 VRAM、便携模型包和其他平台未完成。默认关闭 PE，不进行跨步 DiT K/V 复用。

已有合成 288-token 8 步日志将 DiT block 加载/初始化与计算分开，FP16 每步约 49–54 s 用于加载/准备，对应计算约 2.2–3.2 s。它指向权重生命周期作为优先优化对象，但不能把这一比例直接外推为 1024 全流程比例。下一步先测量权重准备/上传的细分成本和 VAE 工作区，再评估预取、驻留、文本投影复用；新版 KV cache 在后续 PE 另行接入。

复现命令见 [完整 pipeline 复现](../../../docs/REPRODUCE-PIPELINE.md)，实施顺序见 [路线图](../../../docs/ROADMAP.md)。

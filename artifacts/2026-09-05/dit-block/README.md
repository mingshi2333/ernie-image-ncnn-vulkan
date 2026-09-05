# 真实 DiT block 验证：2026-09-05

结论：官方 ERNIE-Image-Turbo 第 0 个 DiT block 的转换与 CPU/Vulkan 数值门槛已经通过。三个静态 token 桶、四种配置，共 12 组比较，每组重复三次输出一致。**权重真实，hidden states 和 shared AdaLN 为固定随机输入，尚未运行完整文生图。**

## 实现与来源

- ncnn：`6a1bf000f363714839a36793addc8c879d3d899e`，上游源码未修改。
- 官方模型：`bc68c81e2a1730a394d5fc9fae70713dee940140`，提取 `layers.0.*` 的 11 个 BF16 张量，有效载荷 436,224,512 字节。
- 参考为锁定 Diffusers 的 `ErnieImageSharedAdaLNBlock`。Python/CUDA/pnnx 版本见 `environment.json`、`requirements-reference.lock` 和各 fixture。
- 各桶导出用的 PyTorch wrapper 与官方 CPU FP32 block 输出逐位一致。使用显式 batch=1、原生 SDPA、4 个 RMSNorm、GELU 门控和 shared AdaLN。
- ERNIE 使用完整重复角度和非交错半向量旋转。标准 ncnn RotaryEmbed 对两半复用同一角度表，不能直接替换。当前导出保留两个半向量的独立 cos/sin 运算。

## 主要数值结果

NRMSE 为相对 L2 误差。阈值在候选执行前写入 fixture：FP32 `2e-5`，FP16 `0.01`，BF16 `0.03`。另有全张量尺度的最大绝对误差门槛，定义见复现说明。它们是合成激活单 block 的工程门槛，不是图像质量门槛。

| Token 数量 | CPU FP32 | Vulkan FP32 | Vulkan FP16 | Vulkan BF16 |
|---|---:|---:|---:|---:|
| 24：16 图像 + 8 文本 | 3.122e-6 | 3.042e-6 | 0.001363 | 0.010810 |
| 288：16 图像 + 272 文本 | 3.079e-6 | 2.922e-6 | 0.001222 | 0.009744 |
| 4160：4096 图像 + 64 文本 | 3.295e-6 | 3.347e-6 | 0.001245 | 0.009973 |

每组含两个 padding 文本位置，288-token 桶覆盖位置编号大于 256。Vulkan 比较使用常驻输入 VkMat，计算层均具备 Vulkan 实现。重复执行相同输入三次的最大差为 0。

后续对同一 288-token fixture 做了官方 PyTorch CUDA 校准，TF32 关闭。其 FP32 / FP16 / BF16 相对 CPU FP32 的 NRMSE 分别为 `1.695e-6 / 0.001142 / 0.009062`。这些后验校准没有用来放宽原先的 ncnn 门槛，不能据此宣称两种运行时完全等价。

## GELU 修复与失败记录

当前上游 [Vulkan GELU shader](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/vulkan/shader/gelu.comp) 使用 tanh 近似，官方 ERNIE 使用 erf 形式。保持其它计算不变的 24-token 对照如下。

| 配置 | NRMSE | 判定 |
|---|---:|---|
| 上游 Vulkan GELU | 2.537e-5 | 未通过原 FP32 门槛 |
| 仅 GELU 回退 CPU | 3.041e-6 | 通过，用于定位差异 |
| 项目 ErnieGELU 保持在 GPU | 3.042e-6 | 通过 |

项目 GPU 实现采用 erf/erfc 数值近似，公式来自 Abramowitz–Stegun 7.1.26，参见 [原书第 299 页](https://personal.math.ubc.ca/~cbm/aands/page_299.htm)。它不是与 libm 逐位相同的实现。对 `[-12,12]` 的两种张量布局共 164,224 个样本，GPU FP32 对双精度 erfc 的最大误差约 `4.175e-7`。CPU FP32、GPU FP32/FP16/BF16 的 8 个核函数场景均通过。

早期失败被保留：

- 最初导出含不支持的 batch 维度提示，后改为全程显式 batch=1，并将不支持的提示升级为导出失败。
- 精简构建遗漏 RMSNorm 共用的 LayerNorm shader 和 Reshape 内部的 Flatten 依赖。后者由崩溃调用栈定位。依赖已补齐。
- pnnx 曾将特殊 RoPE 自动融合成标准 RotaryEmbed。改写两个半向量的表达后拒绝这类融合。
- 第一次显式 GPU 输出下载保留了 packing，布局检查按设计拒绝。下载端现显式请求 pack1，计算过程中保留后端所需 packing。
- 最初 CUDA 参考校准的低精度 attention mask dtype 不匹配而失败，修正为与 query 同 dtype 后重新测量。失败日志保留，最终参考见 `runs/reference-backend-calibration-v2/`。

## 4160-token 的初步运行数据

机器为 Ryzen 7 7745HX、RTX 4060 Laptop 8188MiB。以下为三次单 block 调用，中途只有最终验证输出下载。它不包含模型加载、首次输入上传、其它 35 层、去噪循环、文本编码或 VAE。CPU 与 GPU 的传输路径不同，精度也不同，表中数据不是端到端提速倍数。

| 配置 | 三次调用时间（秒） | Flash / 协作矩阵调用 | 进程最大 RSS |
|---|---|---|---:|
| CPU FP32，4 threads | 4.484 / 4.434 / 4.403 | 不适用 | 2787.5 MiB |
| Vulkan FP32 | 1.249 / 1.197 / 1.179 | 0 / 0 | 676.6 MiB |
| Vulkan FP16 | 0.375 / 0.322 / 0.381 | 3 / 3 | 808.0 MiB |
| Vulkan BF16 | 0.403 / 0.392 / 0.381 | 3 / 3 | 811.7 MiB |

GPU 分支由诊断派生层观察锁定实现中的实际条件，之后调用原上游 SDPA，没有替换 attention 数学。FP32、FP16、BF16 三次运行的整卡显存采样最高分别为 5003 / 2436 / 2427 MiB，起始值分别为 1239 / 1257 / 1257 MiB。采样间隔 100ms，包含其它进程，可能漏过短峰值，不能解释为本进程的精确显存峰值。

## BF16 存储和 host weights

[当前 ModelBin](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/modelbin.cpp) 支持 BF16 段。本项目验证原始 FP32 导出来自 BF16 数值，将 7 个 Gemm 的权重无损打包，4 个 RMSNorm 段保留 FP32。

- 文件大小：`872,449,052 → 436,241,436` 字节。
- 还原后的 FP32 文件流 SHA-256 与原始导出完全一致：`852a78cf0ed54ba7c09ca06211b59fa3dedd5b506ad24280783ce44edd4e814e`。
- 4160-token CPU FP32 和 Vulkan FP16 的实际输出 SHA-256 与使用原始 FP32 文件时分别完全一致。
- 这只改变文件表示。加载时 ncnn 将 BF16 展开，不承诺 RAM/VRAM 减半或加载提速。

相同 BF16 文件、FP16 执行、常驻输入下，主机权重选项的初步比较：

| 权重位置选项 | 三次调用时间（秒） | 进程最大 RSS | 整卡采样最高 / 起始（MiB） |
|---|---|---:|---:|
| 默认 | 0.324 / 0.315 / 0.317 | 779.2 MiB | 2352 / 1166 |
| host memory | 0.359 / 0.346 / 0.347 | 1108.0 MiB | 2007 / 1170 |

两者输出逐位相同。这个小样本显示显存与主机内存/执行时间之间的取舍，需要在多 block 调度中重测，不能直接外推到完整 DiT。

## 验证覆盖与剩余边界

Vulkan 构建 6 个 CTest 项通过，包含 24 个 KV cache 场景及 8 个 GELU 场景。CPU-only 配置独立构建，2 个 CTest 项通过。7 个 Python 契约测试检查校验损坏、静态 shape、版本失配、BF16 无损还原、拒绝有损打包与拒绝未消费的数据。

官方文件通过固定版本 HTTPS Range、长度和本地散列验证，没有验证整个远程分片的散列。原始权重、参考输入/输出及失败尝试留在本地忽略目录，这里保留小型 manifest 和报告。

尚待完成：完整官方 pipeline 参考、真实 pipeline 激活、tokenizer/文本编码、其余 35 个 DiT blocks、前后处理/Euler、VAE、8GB 全模型权重调度和 PNG 输出。没有在此阶段承诺整模型速度、图像质量或移动端能力。

复现入口：[REPRODUCE-BLOCK.md](../../../docs/REPRODUCE-BLOCK.md)。各结果以 `runs/` 下原始 JSON 为准，失败与受控实验分别保留于 `failures/`、`controlled/`，校验清单见 `CHECKSUMS.sha256`。

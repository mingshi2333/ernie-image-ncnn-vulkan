# 2026-09-05：旧 ERNIE 移植与当前 ncnn 的差异

> 后续实现已完成本机 1024 原生生成，最新状态见 [pipeline 报告](../artifacts/2026-09-05/pipeline/README.md) 和 [完整复现](REPRODUCE-PIPELINE.md)。本文保留最初调查或局部验证的范围，文中的待办不代表最新整体状态。

本文保留项目启动时的源码判断。后续已完成第 0 个真实 DiT block、GPU erf GELU、BF16 文件存储和 4160-token 对照，更新结果见 [单 block 报告](../artifacts/2026-09-05/dit-block/README.md)。完整文生图仍未接通。

判断：当前 ncnn 已有可直接使用的原生 KV cache 和专用 allocator，足以简化 ERNIE 提示词增强器的缓存管理。完整文生图的主要优化机会仍在 DiT 的数据流和大权重内存调度。此次同时完成源码审查和无权重算子验证，尚未运行 ERNIE 完整模型。

## 版本依据

| 对象 | 锁定版本 | 说明 |
|---|---|---|
| 当前 ncnn | `6a1bf000f363714839a36793addc8c879d3d899e` | 2026-09-04 上游提交，新项目直接依赖 |
| 已有 ERNIE 移植 | `8dcd6e4411137d8abe92c9d78581c4c96d5182c6` | 检查其源码和模型结构，不以 README 示例图证明精度或速度 |
| 旧项目 ncnn | `f6f734f44d66f469fefee9ee401fd1cb5e3d573e` | 实际为 2026-06-17，比旧项目最后一次源码提交更早 |
| 旧转换模型 | `140a052f7919f279de7f697fa54f33bd1c0cac2b` | 读取 PE decoder / DiT chunks 两份轻量 `.param`，没有读取权重 |

这是对明确版本的比较。当前工作机其他 ncnn 分支上的修改未被当作上游能力。

## 两个多月里实际新增了什么

| 能力 | 旧版本 | 当前上游 | 对本项目的影响 |
|---|---|---|---|
| 原生 SDPA KV cache | 已有 `7=1`。旧 PE 的 26 个 SDPA 都已启用 | 继续提供 | 不应再把“增加 KV cache”当作从零实现的新功能 |
| 缓存容量和 allocator | 无专用 KV allocator。旧 Vulkan 通过 Concat 追加 K/V | 2026-08-17 合入 [#6901](https://github.com/Tencent/ncnn/commit/7f162381ab0461f103239bfe6b345a0d6cce7ee1)，CPU/Vulkan 使用专用缓存接口与容量增长 | 接入会话生命周期，减少历史缓存重分配与复制 |
| CPU attention | 有旧 SDPA 实现 | 2026-09-01 合入 [#6923](https://github.com/Tencent/ncnn/commit/60bfa40044efffbba5d4ccae090a037185942a4d)：x86 Flash Attention 优化、LLM GQA 等 | 旧 PE 强制 CPU，值得首先按同模型、同线程数实测此路径 |
| BF16 模型文件 | 旧运行时已经有 BF16 运算选项 | 2026-08-13 增加 [ModelBin BF16 storage](https://github.com/Tencent/ncnn/commit/c189d88a8da46714e8b52b9b4c317c32a12f16d5) | 转换可以研究保存 BF16 权重，需核实加载和转换临时内存，不能等同于权重常驻显存减少到可容纳 |
| CPU block quantization | 基础较少 | 7 月基础设施、8 月多 CPU 架构优化 | CPU 文本模型可能受益。当前 Vulkan Gemm 对 block-quantized weights 仍明确报不支持，不可直接宣称 Vulkan INT4 全模型可用 |
| Vulkan Flash Attention、host-memory weights | 旧版本已存在 | 当前仍有，启用受设备、精度和 shape 影响 | 两者不应被描述为这次升级才新增。重点是确认实际命中并测量 |

源码：[旧 SDPA Vulkan](https://github.com/Tencent/ncnn/blob/f6f734f44d66f469fefee9ee401fd1cb5e3d573e/src/layer/vulkan/sdpa_vulkan.cpp)、[当前 SDPA Vulkan](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/vulkan/sdpa_vulkan.cpp)、[当前 SDPA CPU](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/sdpa.cpp)、[Gemm Vulkan](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/layer/vulkan/gemm_vulkan.cpp)。

## KV cache 应用在哪一段

| 部分 | 原生自回归 KV cache 的作用 | 可以安全缓存的内容 |
|---|---|---|
| PE 提示词增强器 | 直接适用，先 prefill，再逐 token decode | 每个会话独立的 K/V，按 token 位置追加 |
| 文本编码器 | 每条 prompt 通常只完整运行一次，没有连续生成循环 | 完成后的文本特征，键必须包含模型/tokenizer revision、精度、截断策略和 prompt |
| 图文联合 DiT | 不能直接跨 denoising step 复用 K/V | 原始文本投影、shape/prompt 对应的位置表与 mask、每个 sigma 对应的时间嵌入等不变量 |
| VAE decoder | 不属于逐 token 自回归解码 | 与本项缓存优化无关 |

DiT 每轮都输入变化后的 latent 和时间步，shared AdaLN 改变 block 输入，图文 token 又在 joint attention 中互相影响。因此，哪怕原始 prompt 不变，中间文本 token 的 K/V 也不是跨步常量。[官方 Transformer](https://github.com/huggingface/diffusers/blob/7643c4826609c47755e3da0e5b768e8070468f49/src/diffusers/models/transformers/transformer_ernie_image.py) 与 [官方 pipeline](https://github.com/huggingface/diffusers/blob/7643c4826609c47755e3da0e5b768e8070468f49/src/diffusers/pipelines/ernie_image/pipeline_ernie_image.py) 是判断依据。TeaCache 一类近似复用应独立评估图像质量，不进入精确基线。

当前上游的使用边界也需落实到应用代码：

- `7=1` 开启算子缓存，应用仍负责把缓存输出传给下一次 Extractor。它不是自动管理所有会话的全局缓存。
- CPU 缓存应使用 `extract(..., cache, 1)` 保留私有表示。Vulkan 缓存以 `VkMat` 跨 Extractor 保持在设备上。
- 当前 Vulkan 源码只有设置了 `kvcache_vkallocator` 且 allocator 匹配时，才复用预留容量。只设置序列长度提示不能替代专用 allocator。CPU 默认路径也可复用容量，已通过本次验证。
- 每个会话有独立 cache allocator，不能与 blob allocator 使用同一个对象。输入采用 consume-and-replace 约定。浅拷贝不形成独立分支，allocator 的寿命必须长于全部 cache handles。
- 容量提示不是硬上限。跨版本缓存格式不稳定，升级后应从空缓存开始。当前缓存接口限 batch=1。

依据：[上游缓存接口文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md)、[Extractor API](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/net.h)。

## 旧运行时暴露出的优化机会

所有代码位置均针对 [固定版本的 ernie_image_pipeline.cpp](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.cpp)。以下收益是待测的工程判断，不是已测提速。

| 优先级 | 代码证据 | 建议改进 | 验收方式 |
|---|---|---|---|
| P0 | `generate_image` 中 text encoder、DiT、preprocessor 与后建的 VAE 在同一作用域存活，先加载 DiT 才执行 text encode | 文本编码完成后释放编码器，去噪完成后释放 DiT 再加载 VAE。支持预存 embeddings 时直接跳过文本模型加载 | 分阶段 RAM / VRAM 峰值与成图数值对照 |
| P0 | preprocessor、chunks、finalizer 之间传 `Mat`，chunks 输出回到 CPU 后拷出图片 token，Euler 更新也在 CPU | 设计全设备数据流，跨网络使用 `VkMat`，记录在调用方 `VkCompute` 中。image-token slice 和 Euler update 留在 GPU，只在调试或最终输出时下载 | 记录每步 submit、同步点和上传/下载字节。不是看到 `VkMat` 就假设零拷贝 |
| P0 | 旧 DiT 的 36 个 attention 已经导出成 SDPA，未展开为通用 attention 子图 | 保留 SDPA，并用真实 4096+文本 token、FP16/BF16 检查 Flash Attention 路径与数值 | 一个真实权重 block 对照，以及实际 shader 分支和峰值记录 |
| P0 | DiT 约 8B，仅 16 位权重约 14.9 GiB，超过目标显卡容量 | 文本/DiT/VAE 分阶段加载，并比较 host-memory weights、分 block 调度、VAE tiling。确认不会每步从磁盘重读整套 DiT | 单 block 带宽、整轮峰值、swap 增量和端到端耗时 |
| P1 | `run_pred` 每步重算 RoPE 和 mask，preprocessor 每步处理文本；旧代码已有 temb 向量广播 | 缓存依赖不变的表和投影，保留已有广播。按分辨率、文本长度、有效 token 数、branch 等条件失效 | 逐张量精度不变，减少分配和重复计算 |
| P1 | `PromptEnhancer::load` 强制 `gpuid=-1` 和 FP32；`run_decoder` 未指定 cache extraction type，也没有专用 allocator | CPU 先接新版原生 cache 并测当前 x86 attention。之后增加可选 GPU PE，保留 26 层缓存于设备，合并输出提取和 head 调度 | prefill / decode 分别计时，生成 token 和缓存生命周期对照。不能把当前 CPU PE 的多次 `Mat` 提取直接称为现有 GPU 传输瓶颈 |
| P1 | 旧 PE 的 26 个 SDPA 已保留 32/8 GQA；DiT 使用专用 `ErnieImageRoPE` | 不重复实现已经存在的 GQA。研究 PE 的 RoPE 是否可进一步融合，DiT 特殊 RoPE 必须按官方公式验证 | 单算子及一层误差，不能把标准 RotaryEmbed 当成 DiT 的直接替代 |
| P2 | 旧运行时统一关闭 FP16 arithmetic/storage，以 BF16 选项控制执行 | 建立 FP32 参考、BF16、FP16 的分模块精度策略，优先保持 RoPE 频率、归一化和重要累加稳定 | 实际 36 层、8 步的误差积累和图像质量。探针通过不代表全模型通过 |
| P2 | 当前 CPU block quantization 可用，Vulkan block quantization 有明确缺口 | CPU PE 的低比特权重、DiT Vulkan 量化分别立项，确认算子路径后再选格式 | 权重格式、校准集、量化误差和实际后端记录 |

旧代码已经提供缓存机制、GQA、BF16 选项、DiT SDPA 和调试张量导出。新项目的价值是补齐可复现转换、正确性证据和可量化的数据流/内存改进。

## 本次验证结论与下一步

探针在独立的上游 ncnn checkout 上构建，未修改 ncnn 源码。CPU FP32 以及 Vulkan FP32、FP16、BF16 的 24 个场景检查均通过。32 token prefill + 8 次 decode 的 Vulkan 测试中，启用专用 allocator 后 K、V 各使用一个底层缓冲，默认路径各出现九个底层缓冲。

完整记录和限制见 [artifact report](../artifacts/2026-09-05/README.md)。本次没有做旧 ncnn 二进制与新 ncnn 的速度 A/B，也没有运行完整 PE 或 DiT。下一项工作是锁定可运行的参考 Python 环境、生成真实权重单 DiT block 的对照张量，再连通 GPU 数据流与内存生命周期。

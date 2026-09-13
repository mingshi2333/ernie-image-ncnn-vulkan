# 设计索引

这份索引用来回答“为什么这样实现、依据在哪里、下一步该验证什么”。来源先与当前代码核对，实验结果再写回相应条目。详细架构仍在[代码结构](CODE-STRUCTURE.md)，误差仍以[数值记录](NUMERICAL-RESULTS.md)为准。

## 来源与项目

本轮核对日期：2026-09-13。外部项目按具体提交比较，文章中报告的性能保留原机器、输入和计时范围。

| 来源 | 核对版本 | 可借鉴的设计 | 对应本项目 |
|---|---|---|---|
| [futz12 的 ERNIE 移植](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) / [ncnn #6996](https://github.com/Tencent/ncnn/discussions/6996) | `8dcd6e4` | 已准备的模型下载；PE 一次预填充；主流程容易读懂 | [模型准备](REPRODUCE-PIPELINE.md)、[PE 会话](../src/pe_session.cpp) |
| [everythingfornothing 的 ERNIE 移植](https://github.com/everythingfornothing/ernie-image-ncnn-vulkan/tree/77b4cbbd90bb49723efda95b9009c0bab4f0f6f5) / [ncnn #6998](https://github.com/Tencent/ncnn/discussions/6998) | `77b4cbb` | 按实际文本长度计算；单行 mask；定位 Vulkan mask 广播问题 | [conditioning](../src/conditioning.cpp)、[attention](../src/ernie_attention.cpp)、[形状契约](../src/shape_graph.cpp) |
| [ncnn](https://github.com/Tencent/ncnn/tree/3b7bdba7fc8aea8fd46779533eee027df77c639d) | `3b7bdba` | 原生 attention、KV allocator、pipeline cache | [版本锁](../sources.lock.json)、[构建适配](../cmake/Dependencies.cmake) |
| 本项目 | 本轮起点 `8c3e265`；变更文件身份见下方来源清单 | 权重放置、缓存与预取；显存压力恢复；跨块 GPU 激活；数值对照和框架 CI | [内存执行](MEMORY-EXECUTION.md)、[平台验证](PLATFORM-VALIDATION.md) |

上述项目使用的硬件、精度和功能范围不同，目前没有同机配对结果可以据此排序速度或画质。单行 mask 的方向来自 #6998；实现基于本项目锁定的 ncnn 源码，保留已有的 FP32 补偿、分块和 BF16 兼容处理。

## 概念与实现

| 概念 | 实际含义 | 代码与验证 |
|---|---|---|
| padding mask | 同一行标明哪些 K/V 位置有效。DiT 所有查询共用这一行，不必保存完整方阵 | [常量构造](../src/conditioning.cpp)、[四种 shader 的构建适配](../cmake/derive_sdpa_broadcast.py)、[稠密/广播配对测试](../tests/test_attention_mask.cpp) |
| 查询分块 | FP32 attention 每次处理部分查询，仍访问全部 K/V，控制临时分数矩阵 | [attention](../src/ernie_attention.cpp)、[受限工作区测试](../probes/attention_workspace_probe.cpp) |
| 权重驻留与预取 | 按预算选择位置、保留或准备权重，不复用变化的 DiT 隐藏状态 | [权重会话](../src/weight_session.cpp)、[内存执行](MEMORY-EXECUTION.md) |
| PE 的 KV cache | 自回归时保留前缀 K/V；分块预填充改变一次提交多少输入 token | [PE](../src/prompt_enhancer.cpp)、[缓存会话](../src/pe_session.cpp)、[官方 logits 对照](../tools/validate_pe.py) |
| 文本桶 | 选择可容纳提示词的独立导出图。仅改配置无法证明形状或数值正确 | [独立导出](../tools/export_text_block.py)、[图指纹校验](../tools/rebucket_text.py)、[256-token 导出证据](../artifacts/2026-09-13/design-loop/README.md) |

## 本轮判断与后续

1. **先消除重复 mask。** Vulkan 使用单行广播，CPU 在 attention 调用期间做兼容展开；FP32 分块、原生缓存和精度选择继续使用原有策略。稠密/广播配对通过，64×64 CPU 完整出图的 25/25 张量通过、PNG 最大通道差 1。trace 对比器按广播后的逻辑值检查全部 mask，同时保留实际存储身份。
2. **用 256-token 图验证中间桶。** 相比直接从 64 跳到 2048，它能减少中等长度提示词的 padding。本轮独立导出、真实权重单层及 87-token 提示词的完整 25 层 CPU 文本对照通过；DiT 与模型包验收通过后才启用自动选择。
3. **验证已有 PE 分块候选。** 从单层走到 26 层，用保存的官方 token IDs 和逐步 logits 检查。生产默认仍为单 token 预填充；速度结论需要同机同条件计时。
4. **合并已有模型准备步骤。** 下载器可在文件校验后直接调用原生包校验。公共权重清单需要真实的不可变下载地址，不能用占位 URL 当作交付。

本轮的命令、实际结果和失败记录在[实施记录](../artifacts/2026-09-13/design-loop/README.md)。后续获得新来源或完成实验时，先更新该记录，再调整此处结论。

## 检查与回写

[来源清单](design-sources.json)固定外部比较版本以及本轮涉及的本地源码身份。运行：

```sh
python3 tools/check_design_index.py
```

检查只读：验证本地链接和来源文件散列；发现变化后要求重新核对。它不自动重写结论，也不把链接有效当作技术结论正确。修改本轮源码时，需要连同实验记录一起更新清单。新增结论必须链接到代码、实际输出或明确标记的待验证计划。

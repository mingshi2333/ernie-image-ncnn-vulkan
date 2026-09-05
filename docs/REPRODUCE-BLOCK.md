# 真实 DiT block 的复现

> 后续实现已完成本机 1024 原生生成，最新状态见 [pipeline 报告](../artifacts/2026-09-05/pipeline/README.md) 和 [完整复现](REPRODUCE-PIPELINE.md)。本文保留最初调查或局部验证的范围，文中的待办不代表最新整体状态。

本流程使用官方 ERNIE-Image-Turbo 的第 0 个 DiT block，权重真实，hidden states 和 shared AdaLN 输入为固定随机张量。它验证转换和执行语义，尚不包含 tokenizer、文本编码、完整去噪或 VAE。

## 环境

已验证 Linux x86-64、Python 3.14.7、PyTorch 2.12.1+cu130、Transformers 5.2.0、锁定提交的 Diffusers，以及 `pnnx==20260526`。完整包快照为 `requirements-reference.lock`。这是本机验证环境，不代表所有平台都能直接使用这份 CUDA 依赖快照。

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-reference.lock
git submodule update --init third_party/ncnn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF
cmake --build build -j 4
ctest --test-dir build --output-on-failure
.venv/bin/python -m unittest discover -s tests -v
```

CPU 构建可使用新的 `build-cpu` 目录和 `-DERNIE_ENABLE_VULKAN=OFF`。Vulkan 侧需要可用的驱动和 glslang。CPU-only 配置已独立构建并验证。

## 下载与转换

```sh
.venv/bin/python tools/fetch_component.py --block 0
.venv/bin/python tools/export_dit_block.py --output models/converted/block0-small
.venv/bin/python tools/run_block_matrix.py --model models/converted/block0-small/runtime --output outputs/block0-small
```

下载器只提取该 block 的 11 个张量，BF16 有效载荷为 436,224,512 字节。要求服务器严格返回指定 HTTP Range，校验张量长度、来源版本、每个张量及组合文件的 SHA-256。没有下载完整分片，因此没有验证完整上游分片的散列值。

导出器先与官方 `ErnieImageSharedAdaLNBlock` 比较，再运行 pnnx。当前三个输入桶中，导出用的 PyTorch 实现与官方输出逐位一致。导出会拒绝不支持的算子/维度提示，要求 1 个 SDPA、4 个 RMSNorm，并拒绝将 ERNIE 的特殊位置旋转融合为标准 RotaryEmbed。

`runtime/` 是可运行的模型目录。它将 GELU 替换为项目自有的 erf 形式数值实现，并将位置表的两个 ExpandDims 改为等价的 Vulkan Reshape。pnnx 原始图仍保留。`model.json` 记录所有运行文件及参考数据的校验值，验证器在执行前检查它们。

## 更大的输入桶

```sh
# 位置编号超过 256，含两个 padding 文本位置
.venv/bin/python tools/export_dit_block.py --height 4 --width 4 --text-tokens 272 --valid-text 270 --output models/converted/block0-long-text
.venv/bin/python tools/run_block_matrix.py --model models/converted/block0-long-text/runtime --output outputs/block0-long-text

# 1024x1024 对应的图像 token 网格，4096 + 64 tokens
.venv/bin/python tools/export_dit_block.py --height 64 --width 64 --text-tokens 64 --valid-text 62 --output models/converted/block0-1024
.venv/bin/python tools/run_block_matrix.py --model models/converted/block0-1024/runtime --output outputs/block0-1024 --trace-attention
```

每个桶是静态 shape 的独立导出，不应将 24-token 图直接用于 4160 tokens。当前大桶导出耗时约 3 分 45 秒，导出进程组中单进程最大 RSS 约 6.8 GiB，尚未为其它机器建立资源保证。

`--trace-attention` 仅用于 Vulkan 诊断。它通过一个派生层读取锁定 ncnn 的分支选择状态，计算仍委托给未修改的上游 SDPA 实现。它依赖 ncnn 内部头文件，升级时必须重新核验分支条件。结果中的 `flash_calls` 是实际执行时的诊断，不能由精度选项推断替代。

## BF16 文件存储和主机权重

```sh
.venv/bin/python tools/pack_block_weights.py --model models/converted/block0-1024/runtime --output models/converted/block0-1024/bf16-storage
.venv/bin/python tools/validate_dit_block.py --model models/converted/block0-1024/bf16-storage --output outputs/block0-1024-host --backend vulkan --precision fp16 --device-io --repeat 3 --trace-attention --host-weights
```

打包器只接受能够无损表示为 BF16 的 Gemm 权重，保留 RMSNorm 的 FP32 文件段，检查还原后的整个 FP32 权重流 SHA-256。当前文件从 872,449,052 字节降为 436,241,436 字节。这里的 BF16 是**文件存储格式**，运行时的 `--precision` 单独选择。

ncnn ModelBin 在加载时将 BF16 文件展开为浮点数据。因此，这项变化证明磁盘文件缩小、输入字节数减少，不证明峰值 RAM/VRAM 减半或加载一定更快。

`--device-io` 在一次上传之后保留输入 VkMat，由调用方控制 VkCompute，每轮只为验证下载最终输出。重复执行同一输入必须得到相同输出。整模型各 block 之间的设备连接、Euler 更新和调度尚待实现。

## 判定与记录

- NRMSE 为 `||output-reference||₂ / ||reference||₂`。FP32、FP16、BF16 初始门槛分别为 `2e-5`、`0.01`、`0.03`。
- 最大绝对误差另有 `atol + rtol * max(abs(reference))` 门槛。它是全张量尺度的门槛，不是逐元素 allclose。
- 门槛在 ncnn 候选执行前写入 fixture，属于单 block 工程门槛。后续 CUDA 参考校准不改变这些门槛。真实 pipeline 激活、36 层累积误差和图像质量仍需单独验收。
- `result.json` 记录模型、fixture、执行程序及验证器散列，数值结果、重复一致性、调用时延和进程最大 RSS。非零退出、超时、缺失能力和数值失败不能计为通过。
- NVIDIA 显存采样记录的是整个 device 0 的使用量，含其它进程，100ms 采样可能漏过短峰值。它不是本进程或 allocator 的精确峰值。其它厂商设备没有此采样时仍可执行验证。
- 所有模型、输出和失败尝试保留在被 Git 忽略的目录。使用新的输出路径，不覆盖先前结果。小型报告和校验清单见 `artifacts/2026-09-05/dit-block/`。

# 预转换模型

模型仓库：[akashimio/ERNIE-Image-Turbo-ncnn](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn)。这里发布本项目转换并校验过的运行时模型包，下载后不需要重新转换。

| 包 | 大小 | 内容 |
|---|---:|---|
| 文生图 `turbo/` | 23,271,740,211 字节，约 21.67 GiB | 25 层文本编码器、36 层 DiT、VAE decoder、Tokenizer、32/64/2048 文本图 |
| 可选 PE `pe/` | 7,680,869,430 字节，约 7.15 GiB | 完整 26 层提示词增强模型、原生 KV cache 所需图和独立 Tokenizer |

## 下载文生图模型

在项目根目录执行：

```sh
python3 tools/download_model.py --manifest docs/models/turbo-v1.json \
  --output models/ernie-image-turbo/turbo --verify-with build/linux-vulkan/ernie-image

build/linux-vulkan/ernie-image --model models/ernie-image-turbo/turbo \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --width 1024 --height 1024 --precision fp32 --output outputs/apple.png
```

[turbo-v1.json](turbo-v1.json) 的每个下载地址绑定同一个完整 HF commit，包含文件大小和 SHA-256。下载器只依赖 Python 3 标准库，支持断点续传和已有文件校验；所有文件通过后才执行原生校验。Windows 用户将 `--verify-with` 后面的路径换成自己编译的 `ernie-image.exe`。

使用 Hugging Face CLI 可以直接下载当前版本，同时取得模型卡、许可证和来源记录：

```sh
hf download akashimio/ERNIE-Image-Turbo-ncnn --exclude "pe/*" --local-dir models/ernie-image-turbo
build/linux-vulkan/ernie-image --model models/ernie-image-turbo/turbo --verify-model
```

需要严格复现发布时的字节，请使用上面的固定清单；HF CLI 默认读取仓库 `main`。

## 可选：提示词增强

主包可以直接文生图。需要 PE 自动扩写提示词时，再下载独立包：

```sh
python3 tools/download_model.py --manifest docs/models/pe-v1.json --output models/ernie-image-turbo/pe
build/linux-vulkan/ernie-image --pe-model models/ernie-image-turbo/pe --verify-model

build/linux-vulkan/ernie-image --model models/ernie-image-turbo/turbo \
  --pe-model models/ernie-image-turbo/pe --pe-greedy \
  --prompt 'A red apple on a wooden table.' --width 1024 --height 1024 \
  --precision fp32 --output outputs/apple-pe.png
```

PE 使用单独的原生校验入口，所以这里将下载和 `--pe-model ... --verify-model` 分开执行。也可用 `hf download akashimio/ERNIE-Image-Turbo-ncnn --include "pe/*" --local-dir models/ernie-image-turbo` 补齐这个目录。

## 目录与兼容性

主包的 `manifest.json` 和 `objects/` 组成 schema-3 共享模型。图、权重、Tokenizer 和配置按 SHA-256 保存，清单记录原来的逻辑文件名和来源。程序直接读取这些对象；保持目录和对象名称即可。PE 则保留常规的 `.ncnn.param`、`.ncnn.bin` 及其配置文件。

本次发布复用已有转换结果，没有重新量化或修改权重。两个包上传前都通过完整 Python 和原生包校验，主包 89 个文件、PE 61 个文件。模型仓库的 `provenance.json` 和 `files.json` 记录来源、兼容版本和发布文件的散列。

这两个包用于本项目的运行时，包含项目自定义层。主包支持文生图，未包含图生图 encoder；图生图仍按[单独的模型准备说明](../RUNNING.md#reviewed-image-to-image-package-and-cli)操作。FP32 有更充分的数值对照，BF16 仍为实验选项；各样例的实际误差见[数值记录](../NUMERICAL-RESULTS.md)。

## 存储格式与运行精度

同一主包可以选择 `--precision fp32|fp16|bf16`，不需要下载三份权重。这组选项主要控制 Vulkan DiT 的存储路径，并非把整条管线统一切成同一精度：文本编码、Euler latent、残差和归一化等关键部分保留 FP32，默认 CPU VAE 也保留 FP32 激活及 FP64 统计归约。

主要模型权重沿用官方 BF16 值存储，部分 affine、配置和转换派生数据保留 FP32。这里的 BF16 文件压缩与 BF16 运算是两件事：加载后可以展开用于 FP32 计算；已验证的无损重建也不等于运行时内存减半。本次没有另行发布 INT8、Q6 或 Q4 量化权重。

## 来源与许可

基础模型是 [Baidu ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo/tree/bc68c81e2a1730a394d5fc9fae70713dee940140)，固定 revision 为 `bc68c81e2a1730a394d5fc9fae70713dee940140`。模型按 Apache-2.0 发布，本目录保留原始 [LICENSE](LICENSE-ERNIE-Image) 和本次转换分发的 [NOTICE](NOTICE)。C++ 项目代码的 MIT 许可独立于模型权重许可。

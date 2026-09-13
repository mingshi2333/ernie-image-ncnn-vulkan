---
license: apache-2.0
base_model: baidu/ERNIE-Image-Turbo
pipeline_tag: text-to-image
tags:
  - ncnn
  - vulkan
  - cpp
  - ernie-image
  - text-to-image
---

# ERNIE-Image-Turbo ncnn

[ernie-image-ncnn-vulkan](https://github.com/mingshi2333/ernie-image-ncnn-vulkan) 的预转换模型和程序下载。解压适合系统的程序包，输入提示词即可在本地生成图片；首次运行自动下载模型，无需编译或重新转换。下载脚本使用 Python 标准库，实际推理由 C++、ncnn 和 Vulkan 执行。

![木桌上的红苹果，1376×768，Vulkan FP32](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/9924de97ebe85c540ce09e207142a6efee614be7/assets/apple-1376x768.png)

上图是本项目已有实验的原始输出，使用保存的初始噪声，8 步、Vulkan FP32。[提示词和原图来源](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/7826f7df82972a3396c5af9fa87fb5d605553eb0/docs/images/README.md)可核对；普通 seed 运行不保证逐像素复现这张图。

## 下载即用

**先下载程序 ZIP，完整解压，在解压后的目录运行 `run.py`。** 脚本会按固定版本清单自动下载主模型、校验 SHA-256，再启动原生程序。无需手动选择 block。

| 系统 | 程序下载 | 环境 |
|---|---|---|
| Linux x86_64 | [ernie-image-linux-x86_64.zip](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/main/runtime/ernie-image-linux-x86_64.zip?download=true) · [SHA256](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/main/runtime/ernie-image-linux-x86_64.sha256?download=true) | Ubuntu 24.04 或兼容 glibc 2.39 / GCC 13 C++ 运行库的系统 |
| Windows x64 | [ernie-image-windows-x86_64.zip](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/main/runtime/ernie-image-windows-x86_64.zip?download=true) · [SHA256](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/main/runtime/ernie-image-windows-x86_64.sha256?download=true) | Windows 10/11，附带 MSVC 运行库 |
| macOS Apple Silicon | [ernie-image-macos-arm64.zip](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/main/runtime/ernie-image-macos-arm64.zip?download=true) · [SHA256](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/resolve/main/runtime/ernie-image-macos-arm64.sha256?download=true) | macOS 15，附带 MoltenVK；ad-hoc 签名 |

启动脚本需要 **Python 3.10+**，无需额外 Python 包。Linux/Windows 使用系统 Vulkan 驱动，macOS 使用包内的 MoltenVK。先生成一张 512×512 图片：

```sh
python3 run.py --prompt "A red apple on a wooden table, soft daylight." --output apple.png
```

Windows 将 `python3` 换成 `python`。首次运行下载约 **23.27 GB** 主模型到程序目录的 `models/turbo/`，中断后可重新执行同一命令续传。之后可以断网生成图片。默认 512×512、8 步、seed 42、Vulkan FP16；`--output` 相对于当前终端目录，已有文件会保留，再次生成请换文件名。

```sh
# 改分辨率、切换 FP32，复用已经下载的模型
python3 run.py --prompt-file prompt.txt --width 1376 --height 768 --precision fp32 --output landscape.png

# 竖图、FP16
python3 run.py --prompt "A mountain lake at sunrise." --width 768 --height 1024 --precision fp16 --output portrait.png

# 已经下载了本仓库的 turbo/，直接复用
python3 run.py --model /path/to/ERNIE-Image-Turbo-ncnn/turbo --prompt "A red apple." --output reuse.png

# 只下载、校验，或查看设备
python3 run.py --download-only
python3 run.py --verify-model
python3 run.py --diagnose
```

中文或长提示词可写入 UTF-8 文件，使用 `--prompt-file prompt.txt`。更多参数见 `python3 run.py --help-all`。`--models-dir DIR` 可以更改自动下载位置，`--device cpu` 使用 CPU。

### 分辨率与精度

宽高分别须为 **16..2048** 内的 **16 的倍数**，且总面积不超过 **2,097,152 像素**。512×512、768×1024、1024×1024、1376×768、2048×1024 都可使用这同一套主包。2048×2048 超出当前面积限制；尺寸越大，计算和内存需求也越高。

**FP32、FP16、BF16 共用一套模型，已经包含在本次发布中。** `--precision fp32|fp16|bf16` 选择运行路径，无需下载不同精度的目录。主要权重保留官方 BF16 值，运行时可展开用于 FP32；文本编码、残差、归一化和 Euler latent 等关键部分保留 FP32，默认 VAE 使用 FP32 激活和 FP64 统计。因此精度参数主要针对 Vulkan DiT，并不表示整条管线都使用同一种精度。

FP32 有更充分的数值对照；Vulkan 默认 FP16，CPU 默认 FP32，BF16 仍为实验选项。本次未发布 INT8、Q6、Q4 权重。

### 为什么有 objects 和很多 block？

| 目录 | 用途 | 大小 |
|---|---|---:|
| `runtime/` | 选择一个平台的程序 ZIP，内含启动脚本与下载器 | 见对应文件 |
| `turbo/` | **必需**：文本编码器、36 层 DiT、VAE 解码器、Tokenizer 和共享图 | 23.27 GB / 21.67 GiB |
| `pe/` | **可选**：完整 26 层提示词增强模型 | 7.68 GB / 7.15 GiB |

这些是完整模型的正常组成部分。`turbo/manifest.json` 指向 `objects/` 中的图、权重和配置；PE 的 `block-00` 到 `block-25` 则分别保存 Transformer 层。程序自动按顺序加载。**保留原目录和文件名即可，不要只挑一个 block，也不用自行拼成一个大 `.bin`。**

主包包含独立转换的 32/64/2048 文本图，按实际提示词长度选择，图像尺寸在运行时设置。此包用于文生图，未包含图生图 encoder。

### 可选：提示词增强

在普通生成命令上加 `--with-pe`，脚本会额外下载完整 PE 包：

```sh
python3 run.py --with-pe --pe-greedy --prompt "A small village." --width 1024 --height 1024 --output village.png
```

PE 在 CPU 上执行，使用原生 KV cache。已有 PE 时也可传 `--pe-model /path/to/pe`。

### 手动下载模型 / 自行编译

已经有程序时，可使用 HF CLI 只下载主包；下面的命令不会下载 PE 或其他平台程序：

```sh
hf download akashimio/ERNIE-Image-Turbo-ncnn --include "turbo/*" "LICENSE" "NOTICE" "README.md" "provenance.json" "files.json" --local-dir models/ernie-image-turbo
```

在程序包目录执行 `python3 run.py --model /path/to/models/ernie-image-turbo/turbo --verify-model` 即可检查手动下载的包。源码构建和直接使用 `ernie-image --model ...` 的方法见[项目 README](https://github.com/mingshi2333/ernie-image-ncnn-vulkan#构建与运行)。

## 模型格式与版本

`turbo/` 使用项目的 schema-3 共享格式：`manifest.json` 把各个图、ncnn 权重和配置映射到 `objects/` 中按 SHA-256 命名的文件。保留这些文件名和目录结构即可，程序会直接读取它们。`pe/` 使用独立的 PE 格式，其中包含配套的 `.ncnn.param`、`.ncnn.bin` 和 Tokenizer。两个包都需要本项目的运行时和自定义层。

- 基础模型：[baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)，revision `bc68c81e2a1730a394d5fc9fae70713dee940140`。
- 模型上传前核验使用的运行时代码：[`7826f7d`](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/tree/7826f7df82972a3396c5af9fa87fb5d605553eb0)；ncnn 锁定为 `3b7bdba7fc8aea8fd46779533eee027df77c639d`。
- 原始转换清单记录较早的 ncnn revision `6a1bf000f363714839a36793addc8c879d3d899e`，当前运行时已明确验证兼容；清单保持原样。
- 程序包内 `build-info/runtime.json` 记录该平台的代码、ncnn 版本和原生 CI 框架验证结果；`files.json` 为解压文件的校验清单。
- [provenance.json](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/blob/main/provenance.json) 记录来源和包身份，[files.json](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/blob/main/files.json) 记录本次发布文件的字节数与 SHA-256。

## 验证记录

上传前对两个独立暂存包执行了完整 Python 校验和原生命令 `--verify-model`，覆盖文件清单、大小、SHA-256、来源和图契约。文生图包为 89 个文件，PE 包为 61 个文件；原权重和模型清单没有改动。

本次 Linux 程序 ZIP 已在 RTX 4060 Laptop 上完成默认 512×512、8 步、FP16 出图，生成了正常的红苹果图片；2 个 CPU 线程下，程序记录的生成与写图耗时为 241.19 秒。独立目录运行、运行库加载和模型校验均通过，内存限额触发与 OOM 事件计数均为 0。[本次发布验证](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/artifacts/2026-09-13/runtime-distribution/README.md)保存了原图和命令；这是单次运行记录。

完整出图、逐层误差和内存执行数据见[数值记录](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/7826f7df82972a3396c5af9fa87fb5d605553eb0/docs/NUMERICAL-RESULTS.md)。主要实机环境是 Linux、RTX 4060 Laptop 8GB 和 32GB RAM；三平台 CI 覆盖原生构建和框架测试。部分长提示词及低精度数值对照仍有差异，具体结果保留在记录中。

## 许可

基础模型及本转换包按 Apache-2.0 发布，保留了固定上游版本的 [LICENSE](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/blob/main/LICENSE)，转换与来源说明见 [NOTICE](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn/blob/main/NOTICE)。C++ 运行时代码在 GitHub 仓库单独按 MIT 许可发布。程序 ZIP 内保留了运行时及第三方依赖的许可证；ZIP 不包含模型权重。

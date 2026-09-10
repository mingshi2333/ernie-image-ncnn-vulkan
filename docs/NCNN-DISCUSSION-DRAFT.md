<!--
Published in Tencent/ncnn Discussions, Show and tell, 2026-09-10.
https://github.com/Tencent/ncnn/discussions/6985
Style reference: https://github.com/Tencent/ncnn/discussions/6798 (mingshi2333).
Publication authorized by the user on 2026-09-10, with a Rhino-bird stage-three
annotation. The posted body uses immutable public URLs for the original PNGs.
Repository is PUBLIC; its default branch is codex/surpass-reference.
Posted at 2026-09-10T02:12:15Z. Public links and all three images verified.
Full conversion commands and historical detail: PORTING-WALKTHROUGH.md.
Project creation verified through the GitHub API: 2026-09-05T02:41:27Z.
Earliest local implementation commit: 7bf22ea22ae4622441afc100e28ca72589d6436e,
2026-09-05T05:30:17+03:00. Both records fall on 2026-09-05 locally.
-->

# 【腾讯犀牛鸟2026】ERNIE-Image-Turbo 的 ncnn/Vulkan 实现

> 腾讯犀牛鸟开源人才培养计划第三阶段（Shape with AI · 开源课题实战）项目分享。

这个项目创建于 **2026 年 9 月 5 日**，用 C++ 和 ncnn 实现 ERNIE-Image-Turbo 的本地推理，支持文生图、图生图和可选的提示词增强。模型转换完成后可以离线使用，推理端不需要 Python 或 PyTorch。除了命令行程序，也提供 C++ 接口供其他应用调用。

移植时，我也围绕本地机器的显存和 RAM 做了执行层优化，包括逐块加载、注意力分块、权重缓存、后台预取，以及激活和工作区的预算分配与失败恢复。这篇主要分享这些实现和数值对齐方法。

默认由 CPU 执行文本编码和 VAE，Vulkan 执行 DiT。Turbo 使用 8 步 Euler 去噪、CFG=1，图像 latent 在整个去噪过程中保持 FP32。

- 代码：[ernie-image-ncnn-vulkan](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/tree/codex/surpass-reference)。
- 开发环境：Fedora，Ryzen 7745HX + RTX 4060 Laptop 8GB，32GB RAM。
- 主要版本：ncnn `3b7bdba7`、pnnx `20260526`、Transformers `5.2.0`、Python/Rust Tokenizers `0.22.2`。完整依赖与模型来源记录在 [sources.lock.json](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/sources.lock.json)。

下面是原生程序的实际输出，1376×768、8 步、Vulkan FP32：

![木桌上的红苹果，原生 Vulkan FP32，1376×768](images/apple-1376x768.png)

> A red apple on a wooden table, soft daylight, realistic photo.

| 英文提示词，1024×1024 | 中文提示词，1024×1024 |
|:---:|:---:|
| ![白猫与蓝色茶壶](images/cat-1024.png) | ![雪山、蓝色湖泊与松树林](images/lake-1024.png) |

三张均为已保存实验的原始 PNG，完整提示词、参数和来源见[演示图记录](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/docs/images/README.md)。

## 推理流程

```text
提示词 → 可选 PE → tokenizer / 文本编码 → 文本特征
                                              ↓
初始噪声 → [DiT → Euler 更新] × 8 → 反归一化 / 解包 → VAE → RGB
```

几个组件分别处理文字生成、文本特征和图像去噪，运行时按阶段加载和释放权重：

| 组件 | 结构与作用 |
|---|---|
| 图像文本编码器 | 走 `MistralModel`，执行前 25 层，取 `hidden_states[-2]`，得到每个 token 的 3072 维特征，不做 final norm |
| DiT | 36 层 Transformer，hidden 4096，32 头，head_dim 128，文本和图像做非因果联合注意力 |
| VAE | 将去噪结果还原为 RGB，解码前执行统计量反归一化与 latent 解包 |
| 可选 PE | 完整的 26 层 Ministral3，自回归生成增强提示词，使用独立 tokenizer、chat template 和 KV cache |

以固定 1024×1024、64 个文本槽的包为例，DiT latent 的逻辑形状为 `[128,64,64]`。4096 个图像位置与 64 个文本位置拼接，联合序列长 4160。文本只编码一次，后续各步复用文本特征。

文本桶长、有效 token 数和 DiT 文本槽分别处理。程序按实际分词结果选择文本桶，取有效行，再补齐 DiT 文本槽并构造 mask。source32 共享包仍保留 64 个 DiT 文本槽。

Turbo 的调度与更新为：

```text
s_i         = 1 - i/S                   # S=8, i=0..S-1
sigma_i     = 4*s_i / (1 + 3*s_i)
sigma_S     = 0
timestep_i  = 1000*sigma_i
v_i         = DiT(z_i, text, timestep_i)
z_{i+1}     = z_i + (sigma_{i+1} - sigma_i)*v_i
```

`v_i` 是流速度预测，sigma 递减，Euler 的差值为负。实现保留官方 FP32 linspace 与乘加的求值顺序。去噪结束后，按 128 通道统计量计算 `z*sqrt(variance+1e-5)+mean`，再将每组四通道还原为一个通道的 2×2 空间位置，得到 32 通道 VAE 输入。

## 模型转换与计算精度

模型按文本 block、DiT block、输入/输出头和 VAE 转换。pnnx 处理张量图，C++ 负责条件准备、去噪循环和资源生命周期。每个导出 wrapper 都与锁定的官方实现比较，组件输出和完整生成轨迹分别对拍。

ERNIE 的几处计算需要专门保留：

- **三轴 RoPE。** Q/K 各自 RMSNorm 后再旋转，128 维按 `[32,48,48]` 分给三个轴。文本 j 的位置为 `(j,0,0)`，图像 `(y,x)` 为 `(T,y,x)`，T 是有效文本长度。位置表与 non-interleaved 布局一起适配，频率计算保持 FP32。
- **GELU。** MLP 使用 erf 形式，项目注册 `ErnieGELU`，以目标公式近似 erf。所锁定 ncnn 的 Vulkan GELU 使用 tanh 近似，需要在这里区分。
- **FP32 残差。** 真实文本条件下的残差可能超过 FP16 的有限范围。两个加法点使用 `ErnieResidualAdd` 保留 FP32 skip，归一化临时计算也使用 FP32，再将投影输入转回模型存储精度。
- **注意力归约。** FP32 softmax 分母与 P@V 的长求和使用 Kahan 补偿，减小长序列累加误差。
- **VAE GroupNorm。** 均值和中心方差使用 FP64 归约，其余激活与 affine 运算保持 FP32。独立 1024 VAE 对照的 NRMSE 为 `9.41e-7`。

文本 token IDs 使用与 Python 一致的 Tokenizers 版本和模型文件对齐，C++ 执行文本网络。文本编码取层规则另用完整小模型加 hook 核对。

VAE 采用小图导出与目标空间形状特化，脚本核验固定图身份，再独立比较目标分辨率输出。

从单个真实 DiT block 到文本、36 层 DiT、VAE 和模型包组装的完整命令，放在[分步移植说明](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/docs/PORTING-WALKTHROUGH.md)。

## 本项目的显存与内存优化

这部分由项目自己的 C++ 执行层管理。ncnn 提供图执行、Vulkan buffer 和分配器接口，我在这些接口上实现了 ERNIE 的资源调度，决定何时加载权重、哪些数据留在 GPU、什么时候使用 RAM，以及分配失败后从哪里继续。

首先控制权重驻留。文本编码、DiT 和 VAE 按阶段加载与释放，36 层 DiT 再按 block 流式执行。GPU 命令完成后，当前块不再复用的权重就可以回收，块间激活则保留为 `VkMat`，下一块直接接收 GPU 输出。开启 RAM 缓存时，一部分已准备好的块可以跨去噪步骤复用。

注意力工作区也做了单独处理。项目的 FP32 非 Flash 路径按 query 分块，每次最多处理 128 行，每行仍使用完整 K/V。4160 个位置、32 个头时，单个 FP32 分数缓冲区的大小为：

```text
完整矩阵：32 * 4160 * 4160 * 4 bytes ≈ 2.06 GiB
128 行 Q：32 *  128 * 4160 * 4 bytes = 65 MiB
```

这里缩小的是单个注意力分数缓冲区，计算仍覆盖全部位置。权重、Q/K/V 和其他缓冲区由各自的分配器管理。

CPU VAE 的工作区也作了调整，默认使用 ncnn 直接卷积，关闭 Winograd/SGEMM。已记录的 1024×1024 苹果完整运行，峰值 RSS 从旧路径的 **23.03 GiB 降到 5.82 GiB**。

在这个基础上，项目增加了几项可独立配置的内存策略：

| 机制 | 实现方式 |
|---|---|
| 权重自动放置 | `--dit-weights auto`，组件加载前查询显存预算，结合权重估计和预留空间选择 GPU 或 RAM。RAM 权重仍用于 Vulkan 计算 |
| RAM 权重缓存 | `--dit-cache-mib` 设置容量，跨去噪步骤复用已准备权重，内存压力下回收空闲项 |
| 后台预取 | `--dit-prefetch-mib` 设置预算，后台线程使用独立 Net 提前准备下一块，使 CPU 准备工作与当前块计算重叠，最多提前一块 |
| 激活与工作区放置 | 项目的 `AdaptiveVkAllocator` 通过 `--gpu-memory auto --gpu-spill-mib 2048`，为 DiT 新 buffer 选择设备内存或 GPU 可访问的 RAM，按实际分配量计费 |
| 分配失败恢复 | 每个成功的 Euler 步保存完整 FP32 CPU 检查点，明确分配失败时重建 session，从最近完成的步骤继续，默认最多重试三次 |

权重放置和 RAM 缓存分别在 `weight_placement.cpp`、`weight_session.cpp`，预取接在 `block_sequence.cpp` 的逐块执行循环里。激活与工作区由 `vulkan_memory.cpp` 管理，检查点与恢复集中在 `vulkan_denoise.cpp`。这些策略统一接入 CLI 和 C++ 接口，模型计算代码保留在各自组件中。

缓存与预取默认关闭，启用范围是 Vulkan FP32 的 auto/host 权重路径。缓存和预取准入、host buffer 分配前，都要检查系统余量与进程限制，Linux 还检查当前及上级 cgroup。内存压力下只回收空闲缓存，正在计算的块保持有效，资源释放和共享 Vulkan 队列提交都受同步保护。

恢复时关闭额外缓存和预取，自动模式优先使用 RAM，FP32 非 Flash query 块按 128→64→32→16 缩小。图片尺寸、计算精度、步数和完整 K/V 保持原设置。恢复范围是 DiT 的明确分配错误，设备丢失或同步失败则保留原错误退出。

底层的 buffer、上传和队列错误通过源代码校验后的 ncnn 构建副本传递到执行层。原始子模块保持干净，兼容改动编入 ncnn target。

完整 512×512 FP32 回归中，正常与混合内存配置的 25 份张量和 PNG 均与旧 FP32 逐位相同。混合配置记录了 288 次非 device-local RAM 分配，host buffer 峰值 192 MiB，280 次预取全部被消费。耗时、分配明细和恢复验证记录见[内存执行说明](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/docs/MEMORY-EXECUTION.md)。这些机制主要用于控制驻留和内存压力，速度取决于配置与设备。

## PE 与 ncnn 原生 KV cache

可选 PE 使用 ncnn 原生 KV cache。`PeSession` 为会话持有独立 cache allocator，CPU 侧通过 `extract(..., type=1)` 保留不透明缓存句柄。下一轮消费旧句柄，再接回更新的句柄，重置时先释放缓存，再销毁 allocator。

实际 greedy 对照完成 315 个 token 至 EOS，文本/IDs 一致，每步 logits 通过固定门槛。正常入口按 token 预填充。

图像 DiT 的 hidden states 会随去噪步变化，联合注意力里的文本状态也会更新，因此 DiT 每步重新计算 K/V。PE 的自回归缓存与 DiT 的逐步去噪分别管理。

## 与官方实现的数值对照

参考端使用锁定的官方组件，DiT 采用 CUDA FP32 并关闭 TF32，其余阶段使用 CPU FP32。原生端执行自己的分词、文本编码、DiT 和 VAE。两端使用相同提示词和保存的同一份初始 FP32 噪声，8 步、CFG=1、PE 关闭。

下面摘录 FP32 结果，包括本页三张演示图：

| 样例 | RGB 平均差 MAE | 最大通道差 | 张量检查 |
|---|---:|---:|---:|
| 苹果，512×512 | 0.000361125 | 1 | 25/25 |
| 本页苹果，1376×768 | 0.000617291 | 1 | 25/25 |
| 本页白猫，1024×1024 | 0.002676964 | 1 | 24/25 |
| 本页湖泊，1024×1024 | 0.022625605 | 13 | 21/25 |

MAE 按 `0..255` 的 RGB 通道统计，张量检查采用项目自定的数值容差。512 行来自 9 月 9 日的内存回归，三张演示图来自 9 月 6、7 日保存的实验。FP32 是目前数值对照更充分的路径，BF16 保持实验选项。[完整精度、尺寸与版本结果](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/docs/NUMERICAL-RESULTS.md)保留了 FP16/BF16、长提示词和各阶段数据。

验证工具保存每步预测、latent 和解码结果，也支持用官方输入重放指定步骤和 block。这能把本步计算差异与前面累积的输入差异分开检查。对应工具是 `validate_pipeline.py`、`diagnose_pipeline_step.py` 和 block 探针。

## 代码组织与使用

公共入口是 `ernie::generate`，返回 RGB 像素和运行信息，通过回调报告进度。CLI 处理参数、提示词文件和图片 I/O，公共头文件不暴露 ncnn 类型。

```text
include/ernie/pipeline.h       请求、结果、回调
cli/                          参数、UTF-8 提示词、图片 I/O
src/pipeline.cpp              阶段连接与资源交接
src/text_encoder.cpp          文本网络
src/conditioning.cpp          文本特征、位置与 mask
src/denoiser.cpp               Euler 步进
src/dit.cpp                   单次 DiT 预测
src/block_sequence.cpp        逐块执行与预取
src/pe_session.cpp            PE 原生 KV 会话
src/weight_placement.cpp      权重放置
src/weight_session.cpp        RAM 权重缓存
src/vulkan_memory.cpp         DiT buffer 分配与预算
src/vulkan_denoise.cpp        CPU 检查点与恢复
src/vae.cpp, latent_ops.cpp    VAE 与 latent 变换
tokenizer/                    Rust tokenizer 与模型包验证
tools/, probes/, tests/       转换、诊断与回归
```

Linux 构建需要 C++17、CMake 3.21+、Ninja、Rust/Cargo、libpng、Python 3 和 Vulkan 开发库/驱动：

```sh
git clone --branch codex/surpass-reference --recurse-submodules \
  https://github.com/mingshi2333/ernie-image-ncnn-vulkan.git
cd ernie-image-ncnn-vulkan
cmake --preset linux-vulkan
cmake --build --preset linux-vulkan --target ernie-image
```

模型按[分步说明](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/docs/PORTING-WALKTHROUGH.md)转换并组装为 `models/tutorial/turbo-portable` 后，就可以运行：

```sh
build/linux-vulkan/ernie-image \
  --model models/tutorial/turbo-portable --verify-model

build/linux-vulkan/ernie-image --model models/tutorial/turbo-portable \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --precision fp32 --output outputs/apple.png
```

Vulkan 默认精度是 FP16，上例显式选择 FP32。UTF-8 提示词文件用 `--prompt-file`，共享包用 `--width` / `--height` 指定尺寸，图生图使用带 encoder 的包并设置 `--input` / `--strength`。`--report-json` 保存参数与耗时，`--help` 和 `--help-all` 分别显示常用与完整选项。

外部 C++ 应用通过安装包中的 `find_package(Ernie 0.1.0 EXACT CONFIG REQUIRED)` 和 `ernie::pipeline` 链接同一套生成接口。

## CI 框架验证

GitHub Actions 覆盖 Linux、Windows MSVC 和 macOS Apple Clang 的原生构建与框架验证，内容包括算子、KV cache、CLI、Unicode 路径、模型包校验，以及搬移安装目录后的独立 C++ 调用。

[内存执行版本的五个 CI 作业](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34373124004)全部通过：

| 环境 | 已通过检查 |
|---|---:|
| Linux CPU，读取器 OFF / ON | 各 37 |
| Linux Mesa Vulkan | 57 |
| macOS MoltenVK | 57 |
| Windows MSVC | 37 |

每个作业另有 23 项 HTTP/模型清单检查通过。Linux Vulkan CI 启用固定 SDK 1.4.357.1 的 Khronos validation layer。本机 RTX 4060 的 62 项 Vulkan 测试和 37 项 CPU 测试通过，Vulkan 校验层无错误。工作流与逐项记录见[平台验证文档](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/blob/codex/surpass-reference/docs/PLATFORM-VALIDATION.md)。

感谢 [ncnn](https://github.com/Tencent/ncnn)、[pnnx](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)、[ERNIE-Image](https://github.com/baidu/ERNIE-Image)，以及 [zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan) 和 [futz12 的 ERNIE 移植](https://github.com/futz12/ernie-image-ncnn-vulkan) 提供的组织方式与行为参考。

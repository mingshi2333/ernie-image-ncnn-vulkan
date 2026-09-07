# 架构与代码导航

本项目面向 ERNIE-Image-Turbo，使用浅层目录和按模型组件命名的文件。命令行与 C++ 应用共享一条生成流水线；模型准备、诊断与正式推理各有入口。整体执行图见 [README](../README.md#项目架构)。

## 接口和依赖方向

| 层 | CMake 目标 / 入口 | 职责 |
|---|---|---|
| 应用 | `ernie-image` / `cli/` | 解析参数、读取输入图、保存 PNG/JPEG/BMP/TGA、显示进度和完成报告 |
| 公共接口 | `ernie::pipeline` / `include/ernie/pipeline.h` | `GenerationRequest`、RGB 结果、进度回调、校验和设备诊断 |
| 流水线 | `ernie-pipeline` / `src/pipeline.cpp` | 请求校验、组件调用、阶段释放和运行统计 |
| 模型组件 | `ernie-runtime`、`ernie-pe` | 文本编码、PE 会话、DiT、去噪、图生图和原生算子；VAE 解码实现归入 `ernie-pipeline` |
| 模型与形状 | `ernie-model-package`、`ernie-shape-graph`、`ernie-shape-plan` | 模型包验证、文本来源选择、合法尺寸与内存图实例化 |
| 原生分词桥 | `ernie-tokenizer` / `tokenizer/` | C++ 到 Rust C ABI，复用锁定版本 Tokenizers 和包校验 |
| 运行统计 | `ernie-execution-metrics` | 不依赖 ncnn 的主机侧阶段统计 |

`ernie-image` 链接 `ernie::pipeline` 和 libpng；公共头文件只依赖 C++ 标准库。推理实现依赖 ncnn，调用者不需要使用 `ncnn::Mat`。`ernie::generate` 返回像素与实际请求结果，图像文件由调用方处理；可选 trace 由流水线保存用于诊断。

原生生成不调用 Python。`tools/` 的下载、转换、打包和官方对照属于开发阶段。Rust tokenizer 编译为原生静态库，运行时不需要 Cargo。

日常命令行行为集中在 `cli/options.cpp`：无参数、`-h`、`--help` 显示常用帮助，`--help-all` 展开完整参数；选择 CPU 且未指定精度时填入 FP32，明确指定的不兼容精度仍报错。公共 `GenerationRequest` 的默认值保持不变，C++ 调用者自行指定设备和精度。用户输入错误在模型加载前返回具体原因，模型数学和缓存组件不处理这些规则。

## 模块到源码

| 模块 | 主要文件 | 负责的内容 |
|---|---|---|
| 请求与图像 I/O | `cli/options.*`、`prompt_file.*`、`image_io.*`、`generation_report.*` | 参数到请求的映射、UTF-8、输入缩放、PNG 与完成记录 |
| 请求调度 | `src/pipeline.cpp`、`gpu_context.*` | 生命周期、设备上下文、组件连接、回调 |
| 提示词增强 | `src/prompt_enhancer.*`、`pe_session.*`、`pe_graph.*` | 模板与采样、原生 KV cache、内部候选的分块图 |
| 文本条件 | `src/tokenizer.*`、`text_encoder.*`、`conditioning.*` | 分词桥、文本嵌入与编码、padding、RoPE 和 mask |
| 去噪 | `src/denoiser.*`、`dit.*`、`block_sequence.*`、`latent_ops.*` | Euler 步进、36 层 DiT、分块执行、latent 打包 |
| 图生图与 VAE | `src/img2img.*`、`image_encoder.*`、`vae.*` | 强度与时间步后缀、编码、解码 |
| 模型加载 | `src/model_package.*`、`model_config.*`、`component_files.*`、`model_loading.h` | 包配置、共享路径、图文本和权重加载、stdio/mapped 选择 |
| 形状实例化 | `src/shape_plan.*`、`shape_graph.*` | 目标尺寸、有效 token、padding 与允许修改的图字段 |
| 内存 | `src/weight_placement.*`、`weight_session.*`、`host_memory.*` | 权重放置、可选 RAM 缓存和可用内存查询 |
| 算子与诊断 | `src/ernie_*`、`tensor_io.*`、`execution_metrics.*` | 模型算子、精度处理、诊断张量和计时 |

每个目录自己的 `CMakeLists.txt` 管理本目录目标，根目录管理选项、依赖和安装。原生可执行文件统一输出到所选构建目录的根部。

## 一次生成请求

1. `generate` 检查请求和设备，`ModelPackage` 完整校验运行文件并确定目标尺寸。
2. 图生图请求先执行 VAE encoder；强度为零时直接重建并返回，不进入 PE、文本和去噪。
3. 启用 PE 时，在 CPU 上完成提示词增强，释放 PE 的 26 层权重和 KV cache。
4. 对最终提示词分词，选择可容纳真实 token 的最小可用文本来源，执行文本编码并准备 DiT 条件。
5. 从初始噪声或图生图 latent 出发执行 DiT 和 Euler。文本权重在去噪前释放，DiT 权重在解码前释放。
6. VAE 解码并返回 RGB、实际提示词、token IDs 和运行统计；CLI 或应用决定如何保存、显示结果。

公共入口和字段定义以 [pipeline.h](../include/ernie/pipeline.h) 为准。Vulkan 使用进程级设备上下文，生成调用当前应串行执行。模型文件读取、推理错误通过异常交给调用者；`cli/windows_main.cpp` 统一处理 Windows 命令行字符边界，API 的字符串始终使用 UTF-8。

## 推理数据与计算

### 图像在模型内部是什么形状

文生图从高斯噪声 latent 开始。latent 是模型学习的压缩图像表示，其通道不对应 RGB 颜色。设目标宽高为 `W`、`H`，有效文本长度为 `T`，DiT 为文本预留的槽数为 `L`；联合序列长度为 `N = (H/16) × (W/16) + L`。

下表是逻辑维度，省略 batch=1，不表示 ncnn 的物理 packing/stride。示例取 1024×1024、`L=64`；source32 包即使只有 15 个有效 token，也保留 64 个 DiT 文本槽。

| 数据 | 逻辑形状 | 1024×1024 示例 |
|---|---|---|
| 编码后保留的有效文本 | `[T, 3072]` | 每个有效 token 有 3072 个特征 |
| 去噪主 latent | `[128, H/16, W/16]` | `[128, 64, 64]`，FP32 |
| 输入头展开的图像 token | `[(H/16)×(W/16), 128]` | `[4096, 128]` |
| 图像/文本投影后的联合隐藏状态 | `[N, 4096]` | `[4160, 4096]` |
| DiT 单步预测 | `[128, H/16, W/16]` | 与主 latent 同形状 |
| 反归一化并解包后的 VAE 输入 | `[32, H/8, W/8]` | `[32, 128, 128]` |
| VAE 浮点输出 | `[3, H, W]` | `[3, 1024, 1024]` |
| 对外 RGB 字节 | `[H, W, 3]` | `[1024, 1024, 3]`，交错 uint8 |

文本编码先在选中的静态桶内执行，随后只取有效的 `T` 行；DiT 再按 `L` 补零，并屏蔽 padding key。桶长、有效长度和 DiT 文本槽不是同一个量。实际实现见 [文本准备](../src/pipeline.cpp)、[条件张量](../src/conditioning.cpp)和 [latent 操作](../src/latent_ops.cpp)。

### DiT 如何结合文字与空间位置

输入头把 128 维图像 token 和 3072 维文本特征分别投影到 4096 维，再按“图像在前、文本在后”拼接。每个 block 使用 32 个注意力头，每头 128 维。Q/K 经过各自的 RMSNorm 和 RoPE 后，执行带 padding mask 的全局自注意力：

```text
attention(Q, K, V) = softmax(Q K^T / sqrt(128) + mask) V
```

这里不使用因果 mask，图像与有效文本可以互相读取。前面的文本编码器和可选 PE 才使用因果注意力。DiT 输出头只保留图像位置，并投影回每个位置 128 维的预测。

三轴 RoPE 让注意力区分文本顺序和图像二维位置，128 个旋转维度按 `[32, 48, 48]` 分配。当前 `conditioning.cpp` 中，图像位置 `(y,x)` 的坐标为 `(T,y,x)`，文本位置 `j` 的坐标为 `(j,0,0)`。第一轴使用真实文本长度，不使用补齐桶长。ERNIE 的全宽重复角度表和非交错旋转不能套用“两个半区共享同一角度表”的常规融合；[导出包装](../tools/export_dit_block.py)显式保留这组公式。

每个 block 包含注意力和 gated MLP 两个残差分支。进入分支前先归一化，再施加当前时间条件产生的 scale/shift；分支输出乘 gate 后加回残差。MLP 保留 erf 形式的 GELU。这样同一组学习权重可以在不同噪声阶段产生不同的更新。

### 时间条件与 Euler 步进

`timestep_features` 先生成 4096 维正弦/余弦时间特征。输入头的学习层再得到时间嵌入和 shared AdaLN 的六个 4096 维向量：注意力分支与 MLP 分支各使用一组 shift、scale、gate。这些调制向量在当前步的 36 个 block 间共享，并在下一步重新计算；模型权重本身不会在推理时更新。

当前 Turbo 调度在数学上写作下面的形式，`S` 为步数，`i=0..S-1`：

```text
s_i         = 1 - i/S
sigma_i     = 4*s_i / (1 + 3*s_i)
sigma_S     = 0
timestep_i  = 1000*sigma_i
v_i         = DiT(z_i, text_features, timestep_i)
z_{i+1}     = z_i + (sigma_{i+1} - sigma_i)*v_i
```

`v_i` 表示流速度预测。`sigma` 下降时差值为负，代码按这个符号更新 latent。默认 `S=8`，CFG=1 只调用有条件分支；8 步对应八次完整 DiT，单次 DiT 包含输入头、36 个 block 和输出头。源码对 FP32 linspace 的近端求值、乘除和 Euler 加法顺序有明确处理，数学上相同的改写不一定保留舍入结果。见 [调度与更新](../src/latent_ops.cpp)、[去噪循环](../src/denoiser.cpp)和 [输入/输出头](../tools/export_dit_heads.py)。

### 从 latent 回到 RGB

去噪结束后，每个打包通道先执行 `z * sqrt(variance + 1e-5) + mean`，这里使用模型包中的 128 通道统计量。随后把每组四个通道放回一个通道的 2×2 空间位置，得到 32 通道 VAE 输入。这一步是通道与空间的重排，不是插值放大。

VAE 将压缩表示解码为三通道浮点图。流水线执行 `clamp(decoded/2 + 0.5, 0, 1)`，乘 255 并按 `nearbyint` 转换为 RGB 字节。CLI 再将字节编码为图片文件。因此 PNG 通道误差经过了量化，与浮点 latent 或 decoded 的误差不是同一种指标。

图生图先编码用户图片。非零强度选择同一调度的后缀，以 `(1-sigma)*encoded + sigma*noise` 构造起点；强度决定参与去噪的步数，步数取整后相邻强度可能使用同一后缀。`strength=0` 只做 VAE 重建，`strength=1` 的起点为完整噪声。这是本项目明确实现的强度映射，见 [img2img.cpp](../src/img2img.cpp)。

## 从官方组件到 ncnn 执行

模型准备和原生推理采用不同的执行环境：

```text
准备阶段：固定官方权重 → 组件导出包装 → pnnx 图转换 → 图适配/校验 → 模型包
运行阶段：C++ 请求 → 已准备的 param/bin → ncnn CPU/Vulkan 算子 → RGB
```

导出包装明确输入/输出和 batch=1，保留官方计算公式；pnnx 将受支持的 PyTorch 图转换成 ncnn 层。文本层、DiT 输入头、36 个 block 和输出头分别构图，避免要求整个模型同时驻留。VAE 使用已有转换图及受审查的空间特化；不能把这条路径写成任意高分辨率整图导出都已成功。[转换工具入口](../tools/README.md)

`.param` 描述层、连接与参数，`.bin` 提供学习权重。公共加载边界把图文本和权重路径交给组件；ncnn `Net` 建立算子，`Extractor` 绑定输入并取得输出。对不能直接沿用的 ERNIE 语义，项目注册专用 GELU、残差与归一化层；FP32 注意力在固定 ncnn shader 的完整指纹校验后派生补偿累加实现。

Vulkan 路径使用 `VkMat` 保存设备上的中间状态，通过 `VkCompute` 提交计算。一个块的命令执行完毕后，才能释放其权重；默认流式路径因此在块边界等待完成，并将输出设备张量交给下一块。共享 pipeline cache 复用着色器管线信息，但它不是模型权重缓存。逐块加载降低同时驻留量，代价包含文件读取、权重准备和同步；并不意味着全部加载工作可以直接移到任意后台线程。[块执行](../src/block_sequence.cpp)、[DiT 调度](../src/dit.cpp)

## 模型包和运行时形状

`ModelPackage` 通过 Rust 层验证完整包，再以 `ComponentFiles` 向组件提供图文本和权重路径。组件不解析 JSON 或自行寻找模型包；旧 probes 的目录参数适配到同一个加载边界。

静态 schema-1/2 包保留原有尺寸与容量。schema-3 共享包验证源 manifest，按真实分词结果选择独立的 32/64/2048 文本来源；`ShapePlan` 处理目标尺寸、有效长度和 padding，`shape_graph.cpp` 只修改已知完整图允许变化的字段。图在内存中实例化，不为每个尺寸复制权重或写临时 param 文件。

当前实验范围为每轴 16..2048、16 的倍数、面积最多 2097152。32-token 文本来源保留 64 个 DiT 文本槽，padding 不计入有效文本。图生图还要求该目标尺寸有已验证的 encoder，不能仅凭文本模板推断可用。已有完整尺寸执行结果见 [大尺寸实图报告](../artifacts/2026-09-07/runtime-large/README.md)，其他质量与平台限制见 [验证状态](VALIDATION-STATUS.md)。

## 内存与缓存

- `WeightPlacement` 在加载 DiT 组件前读取 Vulkan 显存预算，结合权重估计与余量选择 GPU 或 RAM；它不持有模型、设备命令或后台线程。
- `block_sequence` 和 DiT heads 管理 Net 的执行及释放。默认逐块加载；中间激活保留在 GPU，激活卸载和分配失败后的自动恢复尚未实现。
- `WeightSession` 管理可选的跨步 RAM 权重缓存，默认容量为零。`host_memory` 提供平台内存余量，Linux 同时考虑 cgroup；不能确认余量时不接纳额外缓存。
- `PeSession` 只管理 PE 的 KV cache、位置和生命周期。它使用 ncnn 的独立 cache allocator、容量 hint 与原生不透明句柄，模板、分词和采样由 `prompt_enhancer` 负责。

PE 的正常入口仍逐 token 预填充。内部候选 `append_chunk` 支持 1..32 个真实 token，通过矩形因果 mask 追加历史；只取最后一个真实 token 的输出进入采样。候选图严格匹配已知图后修改四处序列维度，未改变包和权重。它已有真实单层对照，完整 PE 的收益尚未验证，不作为默认加速说明。

`model_loading.h` 提供 stdio/mapped 读取策略；读取方式、权重放置和缓存容量是独立选项。`cmake/ErnieModelReader.cmake` 的临时缓冲区修正默认关闭，只在构建目录派生固定 ncnn 编译单元，不修改第三方检出。以上开关见 [运行说明](RUNNING.md) 和 [组件说明](REPRODUCE-COMPONENTS.md)。

### 三种缓存的区别

| 机制 | 缓存什么 | 能否精确跨步复用 |
|---|---|---|
| PE KV cache | 自回归历史 token 的 key/value | 在同一因果会话内，追加 token 时可以复用历史 |
| RAM 权重缓存 | 已加载/准备的部分 DiT block 权重与 Net | 权重不随去噪步变化，可以按容量复用 |
| Vulkan pipeline cache | 编译后的着色器管线相关信息 | 可以复用兼容管线，不保存模型的隐藏状态 |

图像 DiT 的联合注意力每一步都会更新图像和文本隐藏状态，其 K/V 不能作为上一去噪步的精确结果直接复用。PE 的历史之所以可缓存，是因果 mask 保证后续 token 不会改变前面 token 的隐藏状态。`PeSession` 使用独立 cache allocator、容量 hint 和 `extract(..., type=1)` 保留 ncnn 原生不透明句柄；缓存交给下一次计算后再接回新句柄，避免把浅拷贝当作独立会话。详见 [pe_session.cpp](../src/pe_session.cpp)。

RAM 权重模式也不表示“显存满了就自动把所有东西转到 RAM”。当前策略在组件加载前选择权重放置位置；激活和 attention 工作区仍需要设备资源，实际分配失败后的恢复尚未实现。

## 模型实现约定

固定 Transformers 版本将官方 Mistral3 配置中的文本子模型分派给 `MistralModel`，不是 `Ministral3Model`。需要的 `hidden_states[-2]` 为 block 24 输出：执行前 25 层，不执行第 26 层或 final norm，也不加载视觉支路和 LM head。该选择同时有完整小模型 hidden-state 钩子检查和真实权重文本路径对照。

DiT 保留三轴 RoPE、erf GELU、shared AdaLN 和最终非 affine LayerNorm。真实文本条件下，残差激活可超过 FP16 的 65504 上限。两个残差相加点使用 `ErnieResidualAdd` 保持 FP32，RMSNorm / LayerNorm 临时计算也使用 FP32，归一化后的投影输入返回模型存储精度。每步 Euler 检查有限值，Vulkan 仅下载 128 个状态浮点数。CPU VAE 的 GroupNorm 使用 FP64 均值与中心方差归约，其余激活和 affine 运算为 FP32。

36 层 DiT 默认每次只加载一块，GPU 中间激活保留在设备上，调用方共享 Vulkan pipeline cache。FP32 注意力对 softmax 分母和概率乘 V 使用 Kahan 累加；查询按最多 128 行处理，每行保留全部 K/V。4160-token、32 头的单个分数矩阵由约 2.06 GiB 降至 65 MiB，代价是增加同步提交；FP16 Flash 和原生 KV cache 路径保留。模型文件的 BF16 表示无损保存官方 BF16 权重，每块约 416MiB，36 块共约 14.63GiB；加载仍会展开和准备权重，文件缩小不代表内存同比缩小。

高分辨率 VAE 使用完整图指纹约束下的两处空间 reshape 特化，并通过独立执行的目标分辨率官方参考。原先整图 pnnx 转换因主机内存持续增长而停止，失败记录保留；不把特化后的成功写成整图导出成功。


## 如何解释数值差异

浮点运算只保留有限精度，归约顺序、乘加融合、存储精度和三角函数实现都会影响结果。即使输入逐位一致，同一公式在 CPU、CUDA 和 Vulkan 后端也不保证逐位相同；这是一般机制，参见 [PyTorch 数值精度说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)。是否存在移植错误仍需独立检查。

一次去噪产生的偏差会成为下一步的输入偏差，再经过注意力、MLP 和归一化传播。文本特征只在开始计算一次，但它影响每一步预测；因此分词完全一致也不等于整个生成过程完全一致。BF16 存储原始官方 BF16 权重可以是无损重编码，而用 BF16 执行中间计算仍可能引入额外舍入。

本项目按四个层次读取证据：token IDs 是否一致；相同输入下组件输出是否接近；从同一保存噪声出发的完整轨迹是否接近；最终 PNG 的通道差是多少。使用官方输入重放某一个 block/步骤可以隔离局部计算，却不能替代原生自由运行的完整轨迹。诊断得到的局部改善也需要完整路径验证后才能推广。

当前 [FP32 中文诊断](../artifacts/2026-09-06/attention-parity/README.md)中，替换为官方文本特征后 PNG 最大差由 13 降到 3，说明文本条件差异会影响该样例，但仍有其他阶段差异。它不是“中文分词有问题”的证据，也不是替换后完成了全原生文本验收。后续 [768 单步隔离](../artifacts/2026-09-07/shared-step768/README.md)和 [相同输入的 block 重放](../artifacts/2026-09-07/shared-blocks17-19-768/README.md)也分别记录了传播误差与局部差异。README 的数值表保留各自来源、配置和失败，不根据这些解释改写旧判定。

## 构建、CI 与维护

构建命令见 [README](../README.md#构建与运行)。安装时启用 `ERNIE_INSTALL_SDK=ON`，独立应用通过 `find_package(Ernie 0.1.0 EXACT CONFIG REQUIRED)` 和 `ernie::pipeline` 接入。安装包带上同版本静态实现库和 ncnn，仍要求兼容的工具链与系统依赖；公共 API 不承诺跨工具链稳定二进制 ABI。

[现有 CI](../.github/workflows/build.yml) 覆盖 Linux CPU/Vulkan 和读取器实验 CPU 构建、无大模型测试、安装与外部消费者。软件 Vulkan 适合小型算子回归，不能代表真实显卡出图。完整模型验证留在本地或有明确硬件的运行环境，不进入每次提交的基础 CI。Windows 的 MinGW/Wine 开发结果见 [Windows 构建说明](BUILDING-WINDOWS.md)，原生 Windows/MSVC/GPU 与 macOS 尚未验证。

新增推理行为先进入对应组件，再由流水线连接，CLI 只增加参数映射。新 UI 使用公共 C++ 接口。转换与诊断分别放在 `tools/` 和 `probes/`，对应测试放在 `tests/`；本地 Linux 交付检查由 `tools/build_release.py`、`check_release.py` 和 `release_dependencies.py` 负责。每次修改检查受影响的行为，完整模型和性能测试按具体问题安排。

历史结果保存在 `artifacts/`；运行成功、数值对照、性能和平台证据各自记录。`models/`、`outputs/`、`build*/` 保持为本地数据目录。工具入口见 [tools/README.md](../tools/README.md)。

## 参考项目与取舍

2026-09-06 核对了以下固定版本的实际目录、构建入口和流水线接口。这里比较的是适用的组织方式，不是开源项目的绝对排名。

| 项目 | 观察到的组织方式 | 本项目采用的部分 |
|---|---|---|
| [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src) | CLI、`ernie_image_pipeline`、tokenizer 放在浅层 `src` 中 | 按推理职责命名文件，能直接找到模型组件 |
| [nihui/zimage-ncnn-vulkan](https://github.com/nihui/zimage-ncnn-vulkan/tree/c1938eef9cd7e03cb216da4f4c89a0c041543e7a/src) | `main`、`zimage_pipeline`、模型实现、`image_io` 和依赖构建文件分开 | 独立流水线、图像文件处理与依赖配置 |
| [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/6b3edaaf32cc19e5bb2d819c788bd557eddc8eba) | `include` 公共接口、`src` 推理、`examples/cli` 和 `examples/server` 应用入口、独立 `cmake` | 公共接口不依赖 CLI，为后续 GUI/服务保留调用边界 |

本项目目前只有 ERNIE-Turbo，不需要引入多模型注册框架或插件层。第三方目录和代码仅作为结构参考，没有复制其运行时实现。

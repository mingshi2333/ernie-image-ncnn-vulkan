# 架构与代码导航

本项目面向 ERNIE-Image-Turbo，使用浅层目录和按模型组件命名的文件。命令行与 C++ 应用共享一条生成流水线；模型准备、诊断与正式推理各有入口。整体执行图见 [README](../README.md#项目架构)。

## 接口和依赖方向

| 层 | CMake 目标 / 入口 | 职责 |
|---|---|---|
| 应用 | `ernie-image` / `cli/` | 解析参数、读取输入图、保存 PNG、显示进度和完成报告 |
| 公共接口 | `ernie::pipeline` / `include/ernie/pipeline.h` | `GenerationRequest`、RGB 结果、进度回调、校验和设备诊断 |
| 流水线 | `ernie-pipeline` / `src/pipeline.cpp` | 请求校验、组件调用、阶段释放和运行统计 |
| 模型组件 | `ernie-runtime`、`ernie-pe` | 文本编码、PE 会话、DiT、去噪、图生图和原生算子；VAE 解码实现归入 `ernie-pipeline` |
| 模型与形状 | `ernie-model-package`、`ernie-shape-graph`、`ernie-shape-plan` | 模型包验证、文本来源选择、合法尺寸与内存图实例化 |
| 原生分词桥 | `ernie-tokenizer` / `tokenizer/` | C++ 到 Rust C ABI，复用锁定版本 Tokenizers 和包校验 |
| 运行统计 | `ernie-execution-metrics` | 不依赖 ncnn 的主机侧阶段统计 |

`ernie-image` 链接 `ernie::pipeline` 和 libpng；公共头文件只依赖 C++ 标准库。推理实现依赖 ncnn，调用者不需要使用 `ncnn::Mat`。`ernie::generate` 返回像素与实际请求结果，图像文件由调用方处理；可选 trace 由流水线保存用于诊断。

原生生成不调用 Python。`tools/` 的下载、转换、打包和官方对照属于开发阶段。Rust tokenizer 编译为原生静态库，运行时不需要 Cargo。

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

## 模型实现约定

固定 Transformers 版本将官方 Mistral3 配置中的文本子模型分派给 `MistralModel`，不是 `Ministral3Model`。需要的 `hidden_states[-2]` 为 block 24 输出：执行前 25 层，不执行第 26 层或 final norm，也不加载视觉支路和 LM head。该选择同时有完整小模型 hidden-state 钩子检查和真实权重文本路径对照。

DiT 保留三轴 RoPE、erf GELU、shared AdaLN 和最终非 affine LayerNorm。真实文本条件下，残差激活可超过 FP16 的 65504 上限。两个残差相加点使用 `ErnieResidualAdd` 保持 FP32，RMSNorm / LayerNorm 临时计算也使用 FP32，归一化后的投影输入返回模型存储精度。每步 Euler 检查有限值，Vulkan 仅下载 128 个状态浮点数。CPU VAE 的 GroupNorm 使用 FP64 均值与中心方差归约，其余激活和 affine 运算为 FP32。

36 层 DiT 默认每次只加载一块，GPU 中间激活保留在设备上，调用方共享 Vulkan pipeline cache。FP32 注意力对 softmax 分母和概率乘 V 使用 Kahan 累加；查询按最多 128 行处理，每行保留全部 K/V。4160-token、32 头的单个分数矩阵由约 2.06 GiB 降至 65 MiB，代价是增加同步提交；FP16 Flash 和原生 KV cache 路径保留。模型文件的 BF16 表示无损保存官方 BF16 权重，每块约 416MiB，36 块共约 14.63GiB；加载仍会展开和准备权重，文件缩小不代表内存同比缩小。

高分辨率 VAE 使用完整图指纹约束下的两处空间 reshape 特化，并通过独立执行的目标分辨率官方参考。原先整图 pnnx 转换因主机内存持续增长而停止，失败记录保留；不把特化后的成功写成整图导出成功。


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

# ernie-image-ncnn-vulkan

ERNIE-Image-Turbo 本地文生图的 C++ / ncnn / Vulkan 实现。原生程序可离线生成图像，推理不依赖 Python；提供命令行和可嵌入应用的 C++ 接口。

当前是实验版本，主要实测环境为 Linux、RTX 4060 Laptop 8GB、32GB RAM、Turbo 8 步、CFG=1、batch=1。完整 1024×1024 离线生成已完成；部分中文、长提示词和逐步数值对照仍有未通过项。Windows 已有 MinGW/Wine 开发验证，原生 Windows/GPU 与 macOS 交付尚待验证。目前不能宣称全面超过参考项目。

[运行说明](docs/RUNNING.md) · [模型准备](docs/REPRODUCE-PIPELINE.md) · [验证结果与限制](docs/VALIDATION-STATUS.md) · [代码结构](docs/CODE-STRUCTURE.md)

## 功能

| 功能 | 当前范围 |
|---|---|
| 原生文生图 | 原生 tokenizer、文本编码、36 层 DiT、8 步去噪与 VAE；完整推理离线执行 |
| 共享权重与运行时尺寸 | 同一包复用权重，按实际 token 数选择独立 32/64/2048 桶；实验范围为轴长 16..2048、16 的倍数、面积 ≤2097152，完整尺寸矩阵仍在验证 |
| 提示词增强 | 可选 CPU PE，使用 ncnn 原生 KV cache；结束后释放 PE 权重与缓存 |
| 图生图 | 已验证的 encoder 包支持图像输入、强度和显式缩放方式；完整质量矩阵仍待验证 |
| 显存与 RAM 权重 | 默认按实时显存预算选择 GPU/RAM；分块加载释放；可选有界 RAM 权重缓存。激活卸载和 OOM 后恢复尚未实现 |
| 输入输出 | UTF-8 提示词文件，PNG/JPEG/BMP/TGA 图像，设备与线程选择，诊断和完成报告 |
| C++ 接口 | 标准库公共头、RGB 输入输出、进度回调；安装后的 `ernie::pipeline` 可供独立应用链接 |
| 模型完整性 | 运行前校验完整文件清单与 SHA256，支持可搬移模型包及旧包入口 |

详细结果和历史失败见[验证状态](docs/VALIDATION-STATUS.md)。完整出图、数值一致性、感知质量、速度和跨平台交付分别验收；各项测量保留版本与输入，不合并成笼统的“全部通过”。

## 构建

需要 C++17、CMake 3.19+、Git、Rust/Cargo、libpng 开发库；Vulkan 构建还需要 Vulkan 开发库和可用驱动。下面使用 ncnn 固定的 glslang 子模块。默认精简 ncnn 层集合已覆盖当前生成器。

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_SYSTEM_GLSLANG=OFF -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF \
  -DERNIE_BUILD_TOKENIZER=ON -DERNIE_BUILD_GENERATOR=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/outputs/install"
```

也可使用系统 glslang 开发库并指定 `-DNCNN_SYSTEM_GLSLANG=ON`。仅 CPU 构建使用独立目录并加 `-DERNIE_ENABLE_VULKAN=OFF`。构建、安装和模型执行的验证范围见[验证状态](docs/VALIDATION-STATUS.md)；Windows 开发要求另见[构建说明](docs/BUILDING-WINDOWS.md)。安装后的二进制仍需要兼容的系统库，详见 [运行说明](docs/RUNNING.md)。

## 转换与生成

完整步骤见 [pipeline 复现](docs/REPRODUCE-PIPELINE.md)。Python 只用于下载、转换、打包和参考验证。模型权重、编译产物及生成图片不进入 Git。组装器先建立开发用链接包，`tools/package_model.py` 再复制实际运行文件，生成不依赖原转换目录的 schema-2 独立包。

本机已验证的模型包可直接运行：

```sh
build/ernie-image --model models/turbo1024-s64-portable --verify-model
build/ernie-image --model models/turbo1024-s64-portable \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --output outputs/apple-new.png --device vulkan --precision fp16 --seed 42 --steps 8
```

输出路径必须不存在，父目录可自动创建。schema-1/2 静态包维持原尺寸和容量；schema-3 共享包使用上述实验性范围，生成时不需要 Python 或为每个尺寸重新转换。多来源共享包必须同时指定 `--width` 和 `--height`。程序在 PE（若启用）结束后按实际 token 数选择可用的最小文本桶；`turbo-shared-v2` 包含独立 32/64/2048 桶，原两来源包只能选择 64/2048。文本桶 32 的来源保留 64 个 DiT 文本槽，padding 不算有效文本。超出可用容量的提示词拒绝，官方分词器的总上限和截断约定保留。CPU 运行需同时指定 `--device cpu --precision fp32`。见[共享包使用方法](docs/RUNNING.md#shared-weights-and-runtime-dimensions)。

长提示词可使用 `--prompt-file UTF8.txt` 替代 `--prompt`，支持可选 BOM、保留原始空白和 CRLF，最多 1 MiB。`--precision bf16` 可运行完整流程，但现有长提示词质量门槛失败，仍标记为实验选项。

本机长提示词独立包为 `models/turbo512x384-s2048-portable`，136 个运行文件、无内部链接，已通过 Python 和原生完整检查。

可选 PE 使用单独的 `models/pe-cpu-v1` 包，通过 `--pe-model` 启用；默认最多生成 2048 tokens，温度 0.6、top-p 0.95，可用 `--pe-greedy` 做确定性对照。PE 权重与缓存会在图像文本编码开始前释放。准备与运行示例见 [运行说明](docs/RUNNING.md)。

每次运行都会在加载权重前检查所有文件，不能用只检查首块或清单文件的方式跳过尾层损坏。`--verify-model` 仅检查模型包，不启动推理。清单用于检测损坏或缺失，不是发布者数字签名。CPU VAE 默认 `--vae-convolution direct`；`sgemm` 保留旧工作区路径以便受控比较。

`--latent FILE.f32` 可指定 FP32 初始噪声；`--embeddings FILE.f32` 可读取与当前提示词 token 数相同的 FP32 文本特征并跳过文本模型。`--trace-dir NEWDIR` 保存实际 token IDs、初始 latent、文本特征、逐步预测及输出。跨框架对照必须读取同一份保存的初始 latent，相同整数 seed 不保证相同噪声。

## 实现中的关键区别

固定 Transformers 版本将官方 Mistral3 配置中的文本子模型分派给 `MistralModel`，不是 `Ministral3Model`。需要的 `hidden_states[-2]` 为 block 24 输出：执行前 25 层，不执行第 26 层或 final norm，也不加载视觉支路和 LM head。该选择同时有完整小模型 hidden-state 钩子检查和真实权重文本路径对照。

DiT 保留三轴 RoPE、erf GELU、shared AdaLN 和最终非 affine LayerNorm。真实文本条件下，残差激活可超过 FP16 的 65504 上限。两个残差相加点使用 `ErnieResidualAdd` 保持 FP32，RMSNorm / LayerNorm 临时计算也使用 FP32，归一化后的投影输入返回模型存储精度。每步 Euler 检查有限值，Vulkan 仅下载 128 个状态浮点数。CPU VAE 的 GroupNorm 使用 FP64 均值与中心方差归约，其余激活和 affine 运算为 FP32。

36 层 DiT 默认每次只加载一块，GPU 中间激活保留在设备上，调用方共享 Vulkan pipeline cache。FP32 注意力对 softmax 分母和概率乘 V 使用 Kahan 累加；查询按最多 128 行处理，每行保留全部 K/V。4160-token、32 头的单个分数矩阵由约 2.06 GiB 降至 65 MiB，代价是增加同步提交；FP16 Flash 和原生 KV cache 路径保留。模型文件的 BF16 表示无损保存官方 BF16 权重，每块约 416MiB，36 块共约 14.63GiB；加载仍会展开和准备权重，文件缩小不代表内存同比缩小。

高分辨率 VAE 使用完整图指纹约束下的两处空间 reshape 特化，并通过独立执行的目标分辨率官方参考。原先整图 pnnx 转换因主机内存持续增长而停止，失败记录保留；不把特化后的成功写成整图导出成功。

## 下一步优化

VAE 直接卷积、64-token 桶和 FP32 查询分块已经完成。后续优先扩大独立提示词与种子数据集，研究已记录的长文本跨步误差累积，处理 DiT 每步重复的权重准备与上传、受控预取，以及按依赖关系复用文本投影和小型条件张量。FP32 分块已在真实模型上降低工作区并完成长英文生成；还需评估更少同步的实现和更多设备，不把整卡采样当作本进程精确显存。冷启动、重复运行和精确 allocator 测量仍待补充。

新版 ncnn 已有原生 KV cache、专用 allocator 和容量管理。本项目已在真实 PE 上验证其 allocator、容量和会话管理；图文联合 DiT 每步的隐藏状态都会变化，跨去噪步复用其 K/V 需要单独的近似算法与质量门槛。CPU PE 已通过固定提示词的完整 greedy 对照；GPU PE、量化、近似缓存和其他平台仍是独立验收项。

- [代码组织、公共接口和依赖方向](docs/CODE-STRUCTURE.md)
- [转换与验证工具入口](tools/README.md)
- [实施路线与验收条件](docs/ROADMAP.md)
- [上游调查和优化依据](docs/2026-09-05-upstream-audit.md)
- [示例项目的精度、内存策略与验证范围](docs/REFERENCE-COMPARISON.md)
- [与 futz12 ERNIE 移植的功能与实际分词对照](docs/FUTZ12-COMPARISON.md)
- [版本、来源与运行时约定](sources.lock.json)
- [项目执行约定](AGENTS.md)

## 来源和许可

本项目新增代码采用 MIT 许可。ncnn 保留 BSD-3-Clause；官方模型及未来引入的第三方代码分别保留原许可和来源。历史参考 [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) 的运行时代码和模型权重未复制到本项目。

缓存 API 依据固定版本的 [ncnn 官方文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md) 和 [会话测试](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/tests/test_sdpa_kvcache_session.cpp)。

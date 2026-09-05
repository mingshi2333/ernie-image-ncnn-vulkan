# ernie-image-ncnn-vulkan

ERNIE-Image-Turbo 本地文生图的 C++ / ncnn / Vulkan 实现。**已从原始提示词生成 1024×1024 PNG**；推理程序不依赖 Python，不访问网络。当前为实验性 Linux 原型：batch=1、Turbo 8 步、CFG=1、提示词增强器（PE）关闭。

2026-09-05，本机 RTX 4060 Laptop 8GB / Ryzen 7745HX / 32GB RAM 实测：英文提示词 “A red apple on a wooden table, soft daylight, realistic photo.” 生成窗边木桌上的红苹果。文本编码使用 CPU FP32，DiT 使用 Vulkan FP16 和 FP32 残差，Euler 主 latent 使用 FP32，VAE 使用 CPU。一次带中间张量记录的运行耗时 **522.17 秒**，进程峰值 RSS **23.03 GiB**，整卡显存 100 ms 采样峰值 **2605 MiB**。整卡数据含其他程序，系统 swap 使用量增加 **0.96 GiB**；这些不是精确的本进程显存或无 swap 基准。

1024 的结果证明本机功能闭环，尚无该分辨率完整官方去噪对照或多提示词质量评估。64×64 的完整链路已经通过预设数值和像素门槛；该小尺寸参考是纹理状数值样例，不能据此判断提示词质量。

| 模块 | 已完成的验证 |
|---|---|
| 原生 tokenizer | C++ + Tokenizers 0.22.2 静态 Rust 库，48 个样本 token IDs 与官方一致 |
| 文本编码器 | 实际 Mistral 文本路径，前 25 层输出；真实英文提示词 CPU NRMSE 6.28e-6，另有中文、空文本对照 |
| 36 层 DiT、8 步 Euler | 合成文本闭环通过 CPU FP32、Vulkan FP32 / FP16；真实文本 FP16 残差溢出已定位并修正 |
| 64×64 完整 prompt → PNG | 同一份保存的初始 latent，对照分阶段执行的官方模块；最终 latent NRMSE 0.05244，PNG 平均误差 1.465/255，全部预设门槛通过 |
| 1024×1024 VAE | 独立官方参考对照，CPU NRMSE 2.02e-6，固定门槛 2e-5 |
| 1024×1024 原生生成 | 36 层、8 步、15 tokens、seed 42，完成 RGB PNG；单次功能和资源实测 |
| 历史独立 DiT block 矩阵 | CPU FP32 / Vulkan FP32 / FP16 各 36/36，BF16 33/36；第 31、33、35 号失败保留 |
| 原生 KV cache | CPU/Vulkan 24 个合成 GQA 场景通过，用于后续 PE / 自回归路径 |

完整证据、固定门槛、失败记录及适用范围见 [pipeline 实测报告](artifacts/2026-09-05/pipeline/README.md)。旧版 [组件报告](artifacts/2026-09-05/components/README.md) 和 [单 block 报告](artifacts/2026-09-05/dit-block/README.md) 保留各自的输入与代码版本，不能与新版本混作同一轮结果。

## 构建

需要 C++17、CMake 3.19+、Git、Rust/Cargo、libpng 开发库；Vulkan 构建还需要 Vulkan 和 glslang 开发环境。默认精简 ncnn 层集合已覆盖当前生成器。

```sh
git submodule update --init third_party/ncnn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF \
  -DERNIE_BUILD_TOKENIZER=ON -DERNIE_BUILD_GENERATOR=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

没有系统 glslang 时，先执行 `git -C third_party/ncnn submodule update --init glslang`，配置时改用 `-DNCNN_SYSTEM_GLSLANG=OFF`。仅 CPU 构建使用独立目录并加 `-DERNIE_ENABLE_VULKAN=OFF`。已验证 Vulkan 构建 17/17 CTest、CPU 构建 6/6 CTest；这些算子与契约检查不需要权重，真实模型对照另行执行。

## 转换与生成

完整步骤见 [pipeline 复现](docs/REPRODUCE-PIPELINE.md)。Python 只用于下载、转换和参考验证。模型权重、编译产物及生成图片不进入 Git；当前模型包由经过验证的本地组件符号链接组成，源目录必须保留，尚不是可分发的独立模型包。

本机已验证的模型包可直接运行：

```sh
build/ernie-image --model models/pipeline1024-residual-v1 \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --output outputs/apple-new.png --device vulkan --precision fp16 --seed 42 --steps 8
```

输出路径必须不存在，父目录需已存在。分辨率由转换时的静态模型桶确定；当前包为 64×64 和 1024×1024。文本编码桶为 **32 tokens（含 BOS）**，超长提示词明确拒绝。PE 关闭，CLI 暂无任意尺寸切换。CPU 运行需同时指定 `--device cpu --precision fp32`。

`--latent FILE.f32` 可指定 FP32 初始噪声；`--embeddings FILE.f32` 可读取与当前提示词 token 数相同的 FP32 文本特征并跳过文本模型。`--trace-dir NEWDIR` 保存实际 token IDs、初始 latent、文本特征、逐步预测及输出。跨框架对照必须读取同一份保存的初始 latent，相同整数 seed 不保证相同噪声。

## 实现中的关键区别

固定 Transformers 版本将官方 Mistral3 配置中的文本子模型分派给 `MistralModel`，不是 `Ministral3Model`。需要的 `hidden_states[-2]` 为 block 24 输出：执行前 25 层，不执行第 26 层或 final norm，也不加载视觉支路和 LM head。该选择同时有完整小模型 hidden-state 钩子检查和真实权重文本路径对照。

DiT 保留三轴 RoPE、erf GELU、shared AdaLN 和最终非 affine LayerNorm。真实文本条件下，残差激活可超过 FP16 的 65504 上限。两个残差相加点使用 `ErnieResidualAdd` 保持 FP32，RMSNorm / LayerNorm 临时计算也使用 FP32，归一化后的投影输入返回模型存储精度。每步 Euler 检查有限值，Vulkan 仅下载 128 个状态浮点数。CPU VAE 的 GroupNorm 使用 FP64 均值与中心方差归约，其余激活和 affine 运算为 FP32。

36 层 DiT 每次只加载一块，GPU 中间激活保留在设备上，调用方共享 Vulkan pipeline cache。模型文件的 BF16 表示无损保存官方 BF16 权重，每块约 416MiB，36 块共约 14.63GiB；加载仍会展开和准备权重，文件缩小不代表内存同比缩小。

高分辨率 VAE 使用完整图指纹约束下的两处空间 reshape 特化，并通过独立执行的目标分辨率官方参考。原先整图 pnnx 转换因主机内存持续增长而停止，失败记录保留；不把特化后的成功写成整图导出成功。

## 下一步优化

当前优先处理 DiT 每步重复的权重准备与上传、受控预取、VAE 工作区内存，以及按依赖关系复用文本投影和小型条件张量。随后扩展文本桶，建立多提示词和完整 1024 官方对照，补充冷启动、重复运行及精确 allocator 测量。

新版 ncnn 已有原生 KV cache、专用 allocator 和容量管理。它对后续 PE 自回归路径有用；图文联合 DiT 每步的隐藏状态都会变化，跨去噪步复用其 K/V 需要单独的近似算法与质量门槛。量化、PE、近似缓存和其他平台都保留为独立验收项。

- [实施路线与验收条件](docs/ROADMAP.md)
- [上游调查和优化依据](docs/2026-09-05-upstream-audit.md)
- [版本、来源与运行时约定](sources.lock.json)
- [项目执行约定](AGENTS.md)

## 来源和许可

本项目新增代码采用 MIT 许可。ncnn 保留 BSD-3-Clause；官方模型及未来引入的第三方代码分别保留原许可和来源。历史参考 [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) 的运行时代码和模型权重未复制到本项目。

缓存 API 依据固定版本的 [ncnn 官方文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md) 和 [会话测试](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/tests/test_sdpa_kvcache_session.cpp)。

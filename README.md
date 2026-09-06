# ernie-image-ncnn-vulkan

ERNIE-Image-Turbo 本地文生图的 C++ / ncnn / Vulkan 实现。**可以离线生成 1024×1024 PNG，已提供独立模型包、文件完整性检查和安装入口。** 推理程序不依赖 Python，不访问网络。当前支持 Linux、batch=1、Turbo 8 步、CFG=1、提示词增强器（PE）关闭，仍是实验版本。

2026-09-05，本机 RTX 4060 Laptop 8GB / Ryzen 7745HX / 32GB RAM 实测：苹果提示词的完整 1024 官方模块对照通过，默认直接卷积将完整生成的峰值进程 RSS 从 **23.03 GiB 降到 5.82 GiB**。新运行耗时 **617.18 秒**，包含约 19.12 秒的全模型散列检查和中间张量记录；这是单次观测，不是受控速度基准。文本、Euler 主 latent 和 VAE 使用 FP32；DiT 默认 Vulkan FP16，并保留 FP32 残差。

**2026-09-06：FP32 注意力已加入补偿累加和查询分块。** 40-token 长英文的完整对照由 20/25 项张量通过提升到 **24/25**，PNG 平均误差由 0.01891 降至 **0.00268/255**、最大差为 **1/255**；最后一步预测的最大张量误差仍超限。中文在补偿累加后的完整运行由 17/25 提升到 **21/25**，PNG 平均误差由 0.03838 降至 **0.02263/255**，最大差 13 仍超过门限 2。苹果历史 FP16 对照通过全部门限，长英文和中文的历史低精度失败保留。这三条固定提示词不构成广泛的感知质量评估。

| 模块 | 已完成的验证 |
|---|---|
| 原生 tokenizer | C++ + Tokenizers 0.22.2 静态 Rust 库，48 个样本 token IDs 与官方一致 |
| 文本编码器 | 实际 Mistral 文本路径，前 25 层输出；独立导出 32/64-token 桶；40-token 真实英文 CPU NRMSE 6.19e-6，另有中文、空文本对照 |
| 36 层 DiT、8 步 Euler | 合成文本闭环通过 CPU FP32、Vulkan FP32 / FP16；真实文本 FP16 残差溢出已定位并修正 |
| 64×64 完整 prompt → PNG | 同一份保存的初始 latent，对照分阶段执行的官方模块；最终 latent NRMSE 0.05244，PNG 平均误差 1.465/255，全部预设门槛通过 |
| 1024×1024 VAE | 直接卷积、独立官方参考，CPU NRMSE 9.41e-7，固定门槛 2e-5；单独解码峰值 RSS 5.42 GiB |
| 1024×1024 苹果完整对照 | 36 层、8 步、15 tokens；最终 latent NRMSE 0.01112、PNG MAE 0.10760/255，全部门限通过 |
| 1024×1024 长英文完整对照 | 40 tokens；最新 FP32 24/25 张量通过，PNG MAE 0.00268、最大差 1，图片通过；整体仍未通过 |
| 1024×1024 中文完整对照 | 32 tokens；最终 FP32 补偿分块版 21/25 张量通过，PNG MAE 0.02263、最大差 13；完整轨迹与未分块补偿版逐位一致，整体未通过 |
| 独立模型包 | 136 个运行文件、约 21.67 GiB；原生 SHA256/文件大小检查，损坏、缺文件、目录搬移检查 |
| 历史独立 DiT block 矩阵 | CPU FP32 / Vulkan FP32 / FP16 各 36/36，BF16 33/36；第 31、33、35 号失败保留 |
| FP32 注意力工作区 | 查询分块与完整修正版在真实 4160-token Q/K/V 上逐位一致；最新长英文整卡 200 ms 采样峰值 4098 MiB，含其他进程 |
| 原生 KV cache | CPU/Vulkan 24 个合成 GQA 场景通过，用于后续 PE / 自回归路径 |

最新证据、固定门槛、失败记录及适用范围见 [注意力改进报告](artifacts/2026-09-06/attention-parity/README.md) 和 [数值诊断说明](docs/NUMERICAL-DIAGNOSTICS.md)。初次交付见 [Turbo 交付报告](artifacts/2026-09-05/turbo-delivery/README.md)。历史 [pipeline 报告](artifacts/2026-09-05/pipeline/README.md)、[组件报告](artifacts/2026-09-05/components/README.md) 和 [单 block 报告](artifacts/2026-09-05/dit-block/README.md) 保留各自的输入与代码版本，不能混作同一轮结果。

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

也可使用系统 glslang 开发库并指定 `-DNCNN_SYSTEM_GLSLANG=ON`。仅 CPU 构建使用独立目录并加 `-DERNIE_ENABLE_VULKAN=OFF`。本机 NVIDIA Vulkan 23/23、CPU 9/9 CTest 和 Python 34/34 通过；这些算子与契约检查不需要权重，真实模型对照另行执行。本地 GCC + 固定 glslang 构建成功，软件 Vulkan 20 项通过、3 项 BF16 因驱动不支持跳过。Linux CPU/Vulkan 工作流已写入，尚未推送触发 GitHub Actions。安装后的二进制仍需要兼容的系统库，详见 [运行说明](docs/RUNNING.md)。

## 转换与生成

完整步骤见 [pipeline 复现](docs/REPRODUCE-PIPELINE.md)。Python 只用于下载、转换、打包和参考验证。模型权重、编译产物及生成图片不进入 Git。组装器先建立开发用链接包，`tools/package_model.py` 再复制实际运行文件，生成不依赖原转换目录的 schema-2 独立包。

本机已验证的模型包可直接运行：

```sh
build/ernie-image --model models/turbo1024-s64-portable --verify-model
build/ernie-image --model models/turbo1024-s64-portable \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --output outputs/apple-new.png --device vulkan --precision fp16 --seed 42 --steps 8
```

输出路径必须不存在，父目录需已存在。分辨率由转换时的静态模型桶确定；已有 64×64 数值测试包和 1024×1024 生成包。文本编码桶为 **32 或 64 tokens（含 BOS）**，上述包为 64。超出所选桶的提示词明确拒绝；更大桶需要独立导出和验证。CLI 暂无任意尺寸切换。CPU 运行需同时指定 `--device cpu --precision fp32`。

每次运行都会在加载权重前检查所有文件，不能用只检查首块或清单文件的方式跳过尾层损坏。`--verify-model` 仅检查模型包，不启动推理。清单用于检测损坏或缺失，不是发布者数字签名。CPU VAE 默认 `--vae-convolution direct`；`sgemm` 保留旧工作区路径以便受控比较。

`--latent FILE.f32` 可指定 FP32 初始噪声；`--embeddings FILE.f32` 可读取与当前提示词 token 数相同的 FP32 文本特征并跳过文本模型。`--trace-dir NEWDIR` 保存实际 token IDs、初始 latent、文本特征、逐步预测及输出。跨框架对照必须读取同一份保存的初始 latent，相同整数 seed 不保证相同噪声。

## 实现中的关键区别

固定 Transformers 版本将官方 Mistral3 配置中的文本子模型分派给 `MistralModel`，不是 `Ministral3Model`。需要的 `hidden_states[-2]` 为 block 24 输出：执行前 25 层，不执行第 26 层或 final norm，也不加载视觉支路和 LM head。该选择同时有完整小模型 hidden-state 钩子检查和真实权重文本路径对照。

DiT 保留三轴 RoPE、erf GELU、shared AdaLN 和最终非 affine LayerNorm。真实文本条件下，残差激活可超过 FP16 的 65504 上限。两个残差相加点使用 `ErnieResidualAdd` 保持 FP32，RMSNorm / LayerNorm 临时计算也使用 FP32，归一化后的投影输入返回模型存储精度。每步 Euler 检查有限值，Vulkan 仅下载 128 个状态浮点数。CPU VAE 的 GroupNorm 使用 FP64 均值与中心方差归约，其余激活和 affine 运算为 FP32。

36 层 DiT 每次只加载一块，GPU 中间激活保留在设备上，调用方共享 Vulkan pipeline cache。FP32 注意力对 softmax 分母和概率乘 V 使用 Kahan 累加；查询按最多 128 行处理，每行保留全部 K/V。4160-token、32 头的单个分数矩阵由约 2.06 GiB 降至 65 MiB，代价是增加同步提交；FP16 Flash 和原生 KV cache 路径保留。模型文件的 BF16 表示无损保存官方 BF16 权重，每块约 416MiB，36 块共约 14.63GiB；加载仍会展开和准备权重，文件缩小不代表内存同比缩小。

高分辨率 VAE 使用完整图指纹约束下的两处空间 reshape 特化，并通过独立执行的目标分辨率官方参考。原先整图 pnnx 转换因主机内存持续增长而停止，失败记录保留；不把特化后的成功写成整图导出成功。

## 下一步优化

VAE 直接卷积、64-token 桶和 FP32 查询分块已经完成。后续优先扩大独立提示词与种子数据集，研究已记录的长文本跨步误差累积，处理 DiT 每步重复的权重准备与上传、受控预取，以及按依赖关系复用文本投影和小型条件张量。FP32 分块已在真实模型上降低工作区并完成长英文生成；还需评估更少同步的实现和更多设备，不把整卡采样当作本进程精确显存。冷启动、重复运行和精确 allocator 测量仍待补充。

新版 ncnn 已有原生 KV cache、专用 allocator 和容量管理。它对后续 PE 自回归路径有用；图文联合 DiT 每步的隐藏状态都会变化，跨去噪步复用其 K/V 需要单独的近似算法与质量门槛。量化、PE、近似缓存和其他平台都保留为独立验收项。

- [实施路线与验收条件](docs/ROADMAP.md)
- [上游调查和优化依据](docs/2026-09-05-upstream-audit.md)
- [示例项目的精度、内存策略与验证范围](docs/REFERENCE-COMPARISON.md)
- [版本、来源与运行时约定](sources.lock.json)
- [项目执行约定](AGENTS.md)

## 来源和许可

本项目新增代码采用 MIT 许可。ncnn 保留 BSD-3-Clause；官方模型及未来引入的第三方代码分别保留原许可和来源。历史参考 [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) 的运行时代码和模型权重未复制到本项目。

缓存 API 依据固定版本的 [ncnn 官方文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md) 和 [会话测试](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/tests/test_sdpa_kvcache_session.cpp)。

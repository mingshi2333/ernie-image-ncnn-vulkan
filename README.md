# ernie-image-ncnn-vulkan

ERNIE-Image-Turbo 本地文生图的 C++ / ncnn / Vulkan 项目。首版目标为 Linux、batch=1、Turbo 8 步、CFG=1，提示词增强器（PE）可关闭。

**当前阶段：36 层 DiT 的小尺寸单次预测已能在 CPU/Vulkan 运行。原生 tokenizer、FP32 Euler 和 latent 解包也已实现，完整文生图尚未接通，当前不能从 prompt 直接生成 PNG。**

截至 2026-09-05，主要结果如下。所有模型验证均使用固定版本的官方权重。

| 模块 | 当前验证范围与结果 |
|---|---|
| ncnn 原生 KV cache | CPU/Vulkan 24 个合成 GQA 场景通过，用于后续 PE/自回归路径 |
| 36 个独立 DiT block | CPU FP32、Vulkan FP32、FP16 各 36/36；BF16 33/36，保留第 31、33、35 号失败 |
| 36 层串联 | 四种配置均通过；FP16 NRMSE 约 0.00270，GPU 中间激活不下载 |
| 输入层 → 36 blocks → 输出层 | 288 tokens 的单次预测，使用保存的合成 latent/text embeddings 与官方时间条件；具体误差见组件报告 |
| 原生 tokenizer | C++ + Tokenizers 0.22.2 静态 Rust 库，48 个样本 token IDs 与官方完全一致 |
| FP32 Euler、BN 与解包 | 21 组 CPU/Vulkan 检查通过，时间步和每步 Euler 输出逐位一致；BN/unpack 在 FP32 误差范围内 |

本项目保留 ERNIE 特有的三轴 RoPE、erf GELU、shared AdaLN 和最终非 affine LayerNorm。GPU GELU 使用 erf 形式；RMSNorm 与 LayerNorm 的临时计算保持 FP32，避免 FP16 平方溢出。36 层按需逐块加载，pipeline cache 由调用方共享；没有跨去噪步复用 DiT K/V。

直接转换器复用经过完整图指纹核验的静态计算图，将每块约 832MiB 的 FP32 文件表示无损缩小为约 416MiB BF16 表示。第 0 块与 pnnx 导出后打包的文件逐字节相同。这是磁盘格式优化，加载仍会展开数据，不代表峰值内存减半。

最新实现、数值门槛、性能对照和负面结果见 [组件与完整 DiT 预测报告](artifacts/2026-09-05/components/README.md)，重跑命令见 [组件复现](docs/REPRODUCE-COMPONENTS.md)。此前的 [单 block 报告](artifacts/2026-09-05/dit-block/README.md) 覆盖 24 / 288 / 4160 tokens，包括 Flash Attention 与协作矩阵分支；其中低精度结果属于归一化修复前的历史版本。

本项目独立组织转换、数值对照和执行代码。已有 [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) 作为历史参考，尚未复制其运行时代码或下载其权重。

## 从这里开始

- [当前技术判断和优化优先级](docs/2026-09-05-upstream-audit.md)
- [实施路线与验收条件](docs/ROADMAP.md)
- [版本与来源](sources.lock.json)
- [第一轮原生缓存实测](artifacts/2026-09-05/README.md)
- [最新组件与完整 DiT 预测实测](artifacts/2026-09-05/components/README.md)
- [历史 DiT 单 block 实测](artifacts/2026-09-05/dit-block/README.md)
- [官方权重下载、转换与复现](docs/REPRODUCE-BLOCK.md)
- [项目执行约定](AGENTS.md)

## 构建与验证

要求 C++17 编译器、CMake 3.19+、Git。Vulkan 构建需要 glslang，可以使用系统安装，也可以初始化 ncnn 的 glslang 子模块。以下 15 项 CTest 检查 KV cache、GELU、RMSNorm/LayerNorm 和 latent 契约，不需要模型权重。真实权重流程另见复现说明。

```sh
git submodule update --init third_party/ncnn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

没有系统 glslang 时，先执行 `git -C third_party/ncnn submodule update --init glslang`，配置时使用 `-DNCNN_SYSTEM_GLSLANG=OFF`。仅构建 CPU 时使用 `-DERNIE_ENABLE_VULKAN=OFF`。

保存可追溯的验证输出：

```sh
python3 tools/run_probes.py --build-dir build --output-dir outputs/probe-run
```

程序输出逐场景 JSON，失败返回非零值。没有 Vulkan 设备时该测试明确跳过，不计作通过。它使用独立的双精度累加参考计算，不把另一条 ncnn 缓存路径当作数值真值。缓存内容被视作后端私有数据，仅观察底层缓冲身份以检查复用，不读取或持久化其内部布局。

默认仅编译探针需要的 ncnn 层。后续完整推理构建必须使用新的构建目录并设置 `-DERNIE_MINIMAL_NCNN=OFF`，或清除旧的 `WITH_LAYER_*` 缓存设置。

## 优化方向

先解决三个问题：可复现的官方权重转换、DiT 去噪过程的数据传输、8GB 显卡上的模型生命周期和权重调度。PE 的新版原生 KV cache 是单独的增强项。

DiT 的图文联合 attention 在每轮去噪中改变，不能直接跨轮复用 K/V。提示词的原始文本特征、位置表、mask 和时间步相关小张量可以按其依赖条件缓存。CPU block quantization 和 Vulkan quantization 需分别验证。

当前完整 DiT 预测使用 4×4 packed-latent 网格与 272 个文本位置（含两个 padding）。1024×1024 的完整 36 层、真实文本编码、8 步 Euler 闭环、VAE 解码、精确峰值 RAM/VRAM 和 PNG 输出仍有独立验收项。

## 来源和许可

本项目新增代码采用 MIT 许可。ncnn 保留其 BSD-3-Clause 许可。官方模型、参考项目及未来引入的第三方代码分别保留原许可和来源。

原生缓存 API 的使用参考了 [ncnn 官方文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md) 与 [会话测试](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/tests/test_sdpa_kvcache_session.cpp)。

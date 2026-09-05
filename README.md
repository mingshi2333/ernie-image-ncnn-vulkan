# ernie-image-ncnn-vulkan

ERNIE-Image-Turbo 本地文生图的 C++ / ncnn / Vulkan 项目。首版目标为 Linux、batch=1、Turbo 8 步、CFG=1，提示词增强器（PE）可关闭。

**当前阶段：真实 DiT 单 block 已通过 CPU/Vulkan 验证，包含 1024×1024 对应的 token 数量。完整模型尚未接通，当前不能生成图片。**

截至 2026-09-05，已完成：

- 锁定 ncnn 上游 `6a1bf000f363714839a36793addc8c879d3d899e`（2026-09-04）。
- 对照旧 ERNIE 移植所用的 2026-06-17 ncnn，记录缓存、精度、内存和执行方式的差异。
- 提供无需模型权重的 C++ 原生 KV cache 验证程序。按 ERNIE 文本模型的 32 Q heads / 8 KV heads / head dimension 128 检查因果 GQA。
- CPU FP32 与 RTX 4060 Laptop 上的 Vulkan FP32 / FP16 / BF16，24 个场景检查通过。覆盖预填充、单 token 追加、多 token 追加、超过容量提示及会话重置。原始数据见 [实测记录](artifacts/2026-09-05/README.md)。
- 从固定版本的官方权重提取并转换第 0 个 DiT block，24 / 288 / 4160 三个 token 桶、四种 CPU/Vulkan 配置共 12 组比较通过，每组重复执行三次输出一致。参考使用真实权重和合成激活。
- 保留 ERNIE 的三轴 RoPE，避开不等价的标准 RotaryEmbed 自动融合。实现 erf 形式的 GPU GELU，修复上游 tanh 近似在该 block 上造成的 FP32 门槛失败。
- 接通常驻输入 VkMat 和调用方 VkCompute，确认低精度大序列实际进入 Flash Attention 与协作矩阵路径。
- 使用 ncnn 原生 BF16 ModelBin 格式，将该 block 权重文件从约 832MiB 无损缩小到 416MiB。还原后的 FP32 权重流校验值和 CPU/Vulkan 输出均一致。

详细结果和未通过的早期尝试见 [真实 block 报告](artifacts/2026-09-05/dit-block/README.md)，完整命令见 [复现说明](docs/REPRODUCE-BLOCK.md)。

本项目独立组织转换、数值对照和执行代码。已有 [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) 作为历史参考，尚未复制其运行时代码或下载其权重。

## 从这里开始

- [当前技术判断和优化优先级](docs/2026-09-05-upstream-audit.md)
- [实施路线与验收条件](docs/ROADMAP.md)
- [版本与来源](sources.lock.json)
- [第一轮原生缓存实测](artifacts/2026-09-05/README.md)
- [真实 DiT block 实测](artifacts/2026-09-05/dit-block/README.md)
- [官方权重下载、转换与复现](docs/REPRODUCE-BLOCK.md)
- [项目执行约定](AGENTS.md)

## 构建与验证

要求 C++17 编译器、CMake 3.19+、Git。Vulkan 构建需要 glslang，可以使用系统安装，也可以初始化 ncnn 的 glslang 子模块。以下 CTest 检查 KV cache 和 GELU，不需要模型权重。真实权重流程另见复现说明。

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

当前结果证明一个真实权重 block 能处理 4096 图像 tokens 加 64 文本 tokens。它不证明完整模型精度、1024×1024 成图、8GB 整模型容量或端到端速度提升。这些均有独立验收项。

## 来源和许可

本项目新增代码采用 MIT 许可。ncnn 保留其 BSD-3-Clause 许可。官方模型、参考项目及未来引入的第三方代码分别保留原许可和来源。

原生缓存 API 的使用参考了 [ncnn 官方文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md) 与 [会话测试](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/tests/test_sdpa_kvcache_session.cpp)。

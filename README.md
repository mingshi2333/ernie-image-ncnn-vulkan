# ernie-image-ncnn-vulkan

ERNIE-Image-Turbo 本地文生图的 C++ / ncnn / Vulkan 项目。首版目标为 Linux、batch=1、Turbo 8 步、CFG=1，提示词增强器（PE）可关闭。

**当前阶段：项目初始化与 attention 验证。尚未接入完整模型，当前不能生成图片。**

截至 2026-09-05，已完成：

- 锁定 ncnn 上游 `6a1bf000f363714839a36793addc8c879d3d899e`（2026-09-04）。
- 对照旧 ERNIE 移植所用的 2026-06-17 ncnn，记录缓存、精度、内存和执行方式的差异。
- 提供无需模型权重的 C++ 原生 KV cache 验证程序。按 ERNIE 文本模型的 32 Q heads / 8 KV heads / head dimension 128 检查因果 GQA。
- CPU FP32 与 RTX 4060 Laptop 上的 Vulkan FP32 / FP16 / BF16，24 个场景检查通过。覆盖预填充、单 token 追加、多 token 追加、超过容量提示及会话重置。原始数据见 [实测记录](artifacts/2026-09-05/README.md)。

本项目独立组织转换、数值对照和执行代码。已有 [futz12/ernie-image-ncnn-vulkan](https://github.com/futz12/ernie-image-ncnn-vulkan/tree/8dcd6e4411137d8abe92c9d78581c4c96d5182c6) 作为历史参考，尚未复制其运行时代码或下载其权重。

## 从这里开始

- [当前技术判断和优化优先级](docs/2026-09-05-upstream-audit.md)
- [实施路线与验收条件](docs/ROADMAP.md)
- [版本与来源](sources.lock.json)
- [第一轮原生缓存实测](artifacts/2026-09-05/README.md)
- [项目执行约定](AGENTS.md)

## 构建与验证

要求 C++17 编译器、CMake 3.19+、Git。Vulkan 构建需要 glslang，可以使用系统安装，也可以初始化 ncnn 的 glslang 子模块。模型权重不参与这个验证。

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

当前算子检查不证明完整模型精度、1024×1024 可运行、8GB 容量可满足或端到端速度提升。这些均有独立验收项。

## 来源和许可

本项目新增代码采用 MIT 许可。ncnn 保留其 BSD-3-Clause 许可。官方模型、参考项目及未来引入的第三方代码分别保留原许可和来源。

原生缓存 API 的使用参考了 [ncnn 官方文档](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/docs/developer-guide/kvcache.md) 与 [会话测试](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/tests/test_sdpa_kvcache_session.cpp)。

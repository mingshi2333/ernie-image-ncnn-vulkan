# 组件、36 层 DiT 与缓存复用验证：2026-09-05

本轮完成全部 36 个官方 DiT block 的直接转换，以及输入投影、时间条件、最终输出层和流式 C++ 执行。小尺寸完整 DiT 单次预测在 CPU FP32、Vulkan FP32/FP16/BF16 上通过预设工程门槛。**文本 embeddings 和初始 latent 为保存的合成输入；当前不能从 prompt 直接生成 PNG。**

原生 tokenizer、FP32 FlowMatch Euler、BN 反归一化与解包也已独立实现。36 块独立测试中的 BF16 有三项失败，原始门槛与失败记录完整保留。`summary.json` 的 `all_blocks_passed=false` 是这一事实的准确表示。

## 版本、输入与范围

- ncnn：`6a1bf000f363714839a36793addc8c879d3d899e`，上游 submodule 未修改。
- 官方 ERNIE-Image-Turbo：`bc68c81e2a1730a394d5fc9fae70713dee940140`。
- Diffusers：`7643c4826609c47755e3da0e5b768e8070468f49`；Transformers 5.2.0；Python/Rust Tokenizers 均为 0.22.2。
- 机器：Ryzen 7 7745HX、约 32GiB RAM、RTX 4060 Laptop 8188MiB。环境详情见 [environment.json](environment.json)。
- 完整 DiT fixture：4×4 packed-latent 网格、272 个文本位置，270 个有效、2 个 padding，共 288 tokens。时间步为 1000。时间正弦特征由官方参考保存，C++ 尚未实现其生成。
- 36 层串联 fixture 使用合成 shared AdaLN；完整 DiT fixture 则由真实输入层与时间 MLP 产生 shared AdaLN。两种条件分别验证，不能合并为同一实验。

NRMSE 定义为相对 L2 误差。所有门槛均在各候选运行前保存：独立 block 的 FP32/FP16/BF16 NRMSE 门槛为 `2e-5 / 0.01 / 0.03`；连接链路为 `0.0002 / 0.03 / 0.12`，另有全张量尺度最大绝对误差门槛。它们是工程诊断门槛，不是感知质量、逐元素 allclose 或最终图像验收。

## 独立 block：141/144 组通过

| 配置 | 通过 | NRMSE 最小值–最大值 |
|---|---:|---:|
| cpu-fp32 | 36/36 | 3.0788e-06–1.08402e-05 |
| vulkan-fp32 | 36/36 | 2.92189e-06–9.00023e-06 |
| vulkan-fp16 | 36/36 | 0.00121939–0.00527357 |
| vulkan-bf16 | 33/36 | 0.00975504–0.0403361 |

BF16 失败如下，索引从 0 开始：

| Block | NRMSE / 上限 | 最大绝对误差 / 上限 | 判定 |
|---|---:|---:|---|
| 31 | 0.031335 / 0.03 | 118.871 / 67.290 | 两项失败 |
| 33 | 0.031898 / 0.03 | 46.792 / 76.516 | NRMSE 失败 |
| 35 | 0.040336 / 0.03 | 126.847 / 121.475 | 两项失败 |

后验官方 CUDA 对照中，block 31 的 BF16 NRMSE 为 0.028223、最大误差 120.075；block 33 为 0.028234、38.831。这说明该输入对低精度敏感，同时 ncnn 仍有额外数值差异。对照没有用于放宽门槛，也没有把失败改为通过。当前以 FP16 作为低显存工作路线，BF16 保持实验状态。

逐块来源、转换图、参考和执行结果见 `blocks/`，两项官方 CUDA 对照见 `runs/reference-block31-calibration/` 与 `runs/reference-block33-calibration/`。

## 36 层连接与完整 DiT 预测

| 配置 | 仅 36 blocks 的 NRMSE | 带前后处理的 NRMSE | 预设门槛 |
|---|---:|---:|---|
| cpu fp32 | 5.2271859e-06 | 1.3676869e-05 | 通过 |
| vulkan fp32 | 5.0809898e-06 | 1.0235286e-05 | 通过 |
| vulkan fp16 | 0.0027044387 | 0.0038261198 | 通过 |
| vulkan bf16 | 0.021596954 | 0.028499875 | 通过 |

全部采用一次加载一个 block 的 stream 策略。输入层结束后释放其 Net；每块提交完成后释放该块权重；输出层随后加载。激活与 shared AdaLN 由调用方 allocator 持有，GPU 中间激活不下载。`peak_loaded_nets=1` 是执行逻辑约束，不是显存计数器。

另有两块 stream/resident 的有界对照，四种精度的输出分别逐位一致。纯 CPU 构建也实际运行了带前后处理的两块前缀，NRMSE 为约 `1.3752e-6`。完整参考与实测见 [串联结果](runs/sequence36-s288-v1/matrix.json)、[单次预测结果](runs/dit36-s288-v1/matrix.json)。

## 实际改进：共享 Vulkan pipeline cache

原先每个 Net 自建并销毁 pipeline cache。现在调用方持有 `ncnn::PipelineCache` 并通过 `Option::pipeline_cache` 传给全部流式 Net，使 pipeline 可跨 block 和头部组件复用。上游 API 和所有权行为见 [net.cpp](https://github.com/Tencent/ncnn/blob/6a1bf000f363714839a36793addc8c879d3d899e/src/net.cpp)。这项缓存与 K/V cache 无关，也没有保留全部模型权重。

两种连接范围 × 三种 Vulkan 精度，共 6 组对照全部通过，所有输出 SHA-256 与旧行为完全相同。完整 DiT 路径中，36 个 block 的加载/初始化阶段为：

| 存储精度 | 每个 Net 独立 cache | 共享 cache | 该阶段时间下降 | 输出 |
|---|---:|---:|---:|---|
| fp32 | 54.53 s | 33.33 s | 38.9% | 逐位一致 |
| fp16 | 161.63 s | 52.23 s | 67.7% | 逐位一致 |
| bf16 | 168.26 s | 52.64 s | 68.7% | 逐位一致 |

这是一次前后观察，权重、输入、精度相同，候选顺序未随机化。该阶段包含 pipeline 创建、模型读取、CPU 权重准备及上传，未细分纯磁盘 I/O；不能据此宣称整张图片提速。仅 blocks 的 BF16 对照下降幅度较小（58.27→51.69 s），也保留在报告中，不能只挑幅度最大的结果。

共享 cache 下，完整 DiT 的阶段合计和资源如下。时间合计包含两端 heads 与 36 块的加载/计算，未覆盖进程启动及最初/最终 I/O，也没有文本编码、8 步采样和 VAE。

| 精度 | 阶段耗时合计 | 进程最大 RSS | 整卡显存采样最高 |
|---|---:|---:|---:|
| fp32 | 37.15 s | 1053.5 MiB | 2399 MiB |
| fp16 | 57.44 s | 1056.3 MiB | 1960 MiB |
| bf16 | 57.48 s | 1057.6 MiB | 1997 MiB |

显存按 100ms 间隔采样，包含其它应用，也可能漏掉短峰值，不是本进程或 allocator 的精确峰值。这里只证明在本机 8GB 显卡上运行了该小尺寸单次预测，不证明 1024×1024 完整生成的容量或速度。小序列下 FP32 的总阶段时间低于 FP16，不作“低精度总是更快”的判断。

对照证据：[仅 blocks](runs/cache-sequence36-s288-v1/matrix.json)、[完整预测](runs/cache-dit36-s288-v1/matrix.json)。

## 修复、转换与其它组件

- **RMSNorm 大值溢出**：原生 FP16 平方临时缓冲超出范围。两块串联的 NRMSE 从 `0.672276` 降到 `0.001470`。修复在设备内将归一化临时运算保持 FP32。
- **LayerNorm 同类问题**：真实输出层、大值合成输入下，FP16 NRMSE 从 `1.000087` 降到 `0.000514`。独立双精度探针同时验证了中心化与非中心化归一化，详见 [复现说明](../../../docs/NCNN-RMSNORM-REPRO.md)。
- **直接 BF16 转换**：复用完整静态图指纹，七个 Gemm 写 BF16 段、四个 RMSNorm 写 FP32 affine。每块 436,241,436 字节；36 块共 15,704,691,696 字节（14.6261 GiB），不含 heads。第 0 块与 pnnx 导出后打包逐字节相同。直接写入约 2.08 s、进程 RSS 约 45MiB，**不含下载、参考生成和首次图导出**，不能称作完整转换速度。见 [散列对照](direct-block00/equality.json)。
- **FP32 latent 运算**：7 套 fixture × CPU、逐步 Vulkan、集中提交 Vulkan，共 21/21 组通过。sigma、timestep 和每一步 Euler 输出与官方 FP32 逐位一致；BN/unpack 的最大绝对误差在 `9.54e-7` 内。纯 CPU 构建另有 7/7。预测与 BN 统计为合成值，尚未接入真实去噪循环。
- **原生 tokenizer**：48 个样本 token IDs 与官方完全一致，非法 UTF-8 明确拒绝；C++ 调用静态 Rust 库，编码不依赖 Python/网络。保留 `ignore_merges=true`，但扫描未找到可区分开关行为的词表反例，不宣称该差异覆盖已经建立。

构建检查：Vulkan 构建 15/15 CTest，纯 CPU 构建 5/5；Python 模型契约 13/13。CTest 内含 24 个 cache 场景、8 个 GELU 场景、16 个 RMSNorm 场景、8 个 LayerNorm 场景及 latent 契约。

## 保留的失败与剩余工作

早期 Vulkan 输入层在卷积创建 pipeline 时崩溃，由调用栈定位到精简构建遗漏 Padding 依赖。第一次输出层导出出现 batch 轴告警；等价改写为始终保持 batch=1 后通过。其它失败还包括：初版 GPU latent 上传 packing 不符合约定；tokenizer 测试错误要求必须找到 ignore_merges 反例；批处理时重建导致执行文件暂时缺失；长 HTTP Range 连接截断。分别修正为显式布局、如实记录未覆盖项、执行文件快照，以及严格长度校验的 32MiB 窗口与有限重试。

失败记录分别位于 `failures/`、`batches/` 和 `logs/`。带数值失败继续扫描的批次仍返回非零状态，不冒充全部通过。旧执行结果与最新源文件的差异通过各自 runner/source SHA 和父提交追溯；本报告的最新源文件清单见 [source-sha256.json](source-sha256.json)。

下一步仍需实现和验证：Mistral3 文本路径及 `hidden_states[-2]`、C++ 时间特征、真实预测的 8 步 Euler、VAE decoder、PNG CLI、真实文本条件与高分辨率的质量/资源边界。调度器的 FP16/BF16 舍入顺序也未实现。后续性能重点是权重准备与跨去噪步生命周期，其次才是归一化融合和更多近似优化；PE 的原生 KV cache 留待完整主链路之后接入。

重跑入口：[组件复现](../../../docs/REPRODUCE-COMPONENTS.md)。此目录只含小型 manifest、图参数和日志，不含权重、执行文件或浮点张量；完整数据留在本地忽略目录。校验清单为 `CHECKSUMS.sha256`。

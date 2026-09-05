# Turbo 本地交付与 1024 对照

日期：2026-09-05。范围：Linux、ERNIE-Image-Turbo、batch=1、8 Euler 步、CFG=1、PE 关闭。

原生提示词 → tokenizer → 25 层文本编码 → 36 层 DiT × 8 步 → BN/unpack → CPU VAE → PNG 已完整运行。独立模型包、运行前完整性检查、安装入口和回归验证已落地。1024 苹果对照通过全部门限；长英文和中文的精度失败按原始结果保留，不能将可用成图等同于全部数值门限通过。

## 交付变更

| 变更 | 实现与证据 |
|---|---|
| CPU VAE 内存 | 默认直接卷积，保留显式 SGEMM；GroupNorm 继续使用 FP64 归约，激活和 affine 为 FP32 |
| 64-token 文本桶 | 独立导出目标图；规范化仅覆盖 13 处已审查 token 参数，完整图指纹与重建 FP32 权重散列必须匹配 |
| 可移动模型包 | 136 个运行文件，23,271,870,723 字节（约 21.67 GiB），没有内部或外部依赖链接；32/64-token 包分别保存 |
| 原生校验 | Rust SHA256 通过 C ABI 提供给 C++；所有文件散列/字节数、版本、配置和清单一致性检查，兼容旧 schema-1 开发包 |
| 安装和构建 | 安装 ernie-image、使用说明和来源锁；Linux CPU/Vulkan 工作流已写入，本轮没有远程 Actions 运行 |
| 官方参考 | 固定官方权重与代码，单块 CUDA FP32 DiT，禁用 TF32，其他模块 CPU FP32；子进程退出后再启动 Vulkan |
| 证据封存 | 校验参考、native 张量、PNG、runner 和验证脚本散列，另用 NumPy 重算全部误差、门限和 PNG 量化 |

清单用于检测文件缺失和损坏，不是发布者数字签名。模型包可搬移，安装二进制仍要求兼容的 Linux 系统库。

## 完整图像对照

每项都使用该项官方参考保存的同一份初始 latent。不同框架的相同整数 seed 不作为等价输入证明。各行是不同提示词/精度的单次结果，不能据此比较性能或宣称广泛质量通过。

| 样本 | tokens / 桶 | 精度 | 张量通过 | PNG MAE / 255 | 最大像素差 / 255 | PNG 门限 | 全部门限 |
|---|---:|---|---:|---:|---:|---|---|
| [apple-fp16](runs/pipeline1024-portable-direct-v1/result.json) | 15 / 32 | fp16 | 25/25 | 0.10760276 | 22 | 通过 | 通过 |
| [long-fp16](runs/pipeline1024-long-s64-v1/result.json) | 40 / 64 | fp16 | 22/25 | 0.53214264 | 79 | 通过 | 未通过 |
| [long-fp32](runs/pipeline1024-long-s64-fp32-v1/result.json) | 40 / 64 | fp32 | 20/25 | 0.01891295 | 2 | 通过 | 未通过 |
| [chinese-fp16](runs/pipeline1024-chinese-s64-fp16-v1/result.json) | 32 / 64 | fp16 | 20/25 | 0.99999905 | 145 | 未通过 | 未通过 |
| [chinese-fp32](runs/pipeline1024-chinese-s64-fp32-v1/result.json) | 32 / 64 | fp32 | 17/25 | 0.03838348 | 23 | 未通过 | 未通过 |

每项 native float → PNG 量化均单独核对。详细指标、失败张量和输入配置见 [results.json](results.json)。

| 样本 | 最终 latent NRMSE | 解码 NRMSE | 总耗时（秒） | 最大 RSS（GiB） |
|---|---:|---:|---:|---:|
| apple-fp16 | 0.011115472 | 0.003211986 | 617.180 | 5.8217 |
| long-fp16 | 0.046383542 | 0.021651909 | 641.168 | 5.8305 |
| long-fp32 | 0.001425029 | 0.000417023 | 704.743 | 5.8154 |
| chinese-fp16 | 0.038761581 | 0.033852030 | 563.846 | 5.8244 |
| chinese-fp32 | 0.001850184 | 0.001632714 | 675.481 | 5.8117 |

提示词如下；每条参考及其 precision 重试均保留原始记录。

- A red apple on a wooden table, soft daylight, realistic photo.
- A small white cat sitting beside a blue ceramic teapot on a wooden desk, warm afternoon sunlight through a window, a green plant in the background, soft shadows, realistic photograph with fine detail.
- 雪山脚下的蓝色湖泊，松树林，清晨阳光，写实风景摄影。

### 固定门限与负结果

门限在运行前写入各 run 的 gates.json，没有根据本轮结果放宽。FP16 张量要求 NRMSE ≤ 0.15，最大绝对差 ≤ `0.03 + 0.25 × reference_max_abs`；FP32 分别要求 ≤ 0.003 和 `0.0002 + 0.01 × reference_max_abs`。FP16 PNG 要求 MAE ≤ 12、最大差 ≤ 80；FP32 为 MAE ≤ 0.1、最大差 ≤ 2，单位均为 8-bit 像素值。

文本/位置条件有独立门限；初始 latent 必须逐位相同。25 项张量中任何一项、PNG 比较或 PNG 量化失败，整体即失败。不能仅用平均像素误差较小将失败覆盖。

- long-fp16：张量失败为 `prediction-6`, `prediction-7`, `decoded`；PNG 通过。
- long-fp32：张量失败为 `prediction-6`, `prediction-7`, `step-7`, `final`, `unpacked`；PNG 通过。
- chinese-fp16：张量失败为 `prediction-3`, `prediction-5`, `prediction-6`, `prediction-7`, `decoded`；PNG 未通过。
- chinese-fp32：张量失败为 `prediction-4`, `prediction-5`, `prediction-6`, `prediction-7`, `step-7`, `final`, `unpacked`, `decoded`；PNG 未通过。

### 误差定位

长英文的第七次预测（index 6）使用官方该步输入 latent、文本、时间特征、RoPE 和 mask 单独执行。FP32 NRMSE 为 7.4248e-5，最大差 0.003621，通过原有张量门限；自由运行同一步的 NRMSE 为 0.0034907，未通过。该结果支持“前序轨迹及条件的小差异被后续运算放大”的解释，尚未隔离到某一个算子或舍入来源。单步通过不能替代自由运行通过。证据见 [单步诊断](components/diagnostic-long-step6-fp32-v1/native/result.json)。

时间特征包含全部 8 个 Turbo 时间点，41 个有效输入与 5 个非法输入均通过原有门限；最大 NRMSE 1.2245e-6。没有找到第七步特有的时间特征异常。

## 内存与组件验证

完整苹果生成最大 RSS 从旧 SGEMM 的 24,145,308 KiB（23.03 GiB）降至直接卷积的 6,104,472 KiB（5.82 GiB），约下降 74.7%。独立 1024 VAE 的最大 RSS 为 5,685,048 KiB（5.42 GiB），NRMSE 9.4142e-7，仍使用固定 2e-5 门限。直接卷积与高精度 GroupNorm 同时保留；不能只关闭 Winograd 后宣称归一化问题已解决。

32/64-token 静态图的规范化完整指纹为 `67881d48031e178fcb5653f749d17895a9470d03adf5bf8098dd5bbff952ab36`。64-token 桶的 40-token 真实文本、25 层 CPU NRMSE 为 6.1913e-6。第一次 rebucket 调用了仅含 block zero 的目录，失败日志保存在 failures/，最终包来自完整 25 层的成功重试。

旧苹果运行耗时 522.17 秒，新直接卷积运行 617.18 秒，后者增加了约 19.12 秒全模型校验；两者都包含 trace。不是受控性能实验，不宣称速度提升。FP32 单步诊断整卡显存 100 ms 采样峰值 7609 MiB，含其他进程；FP32 attention 当前构造完整注意力矩阵，显存成本高于 FP16，尚无精确 allocator 峰值。

## 构建与回归

- 原生 CPU：8/8 CTest 通过。
- 原生 NVIDIA Vulkan：19/19 CTest 通过，详见 logs/ctest-delivery-hardware-final.log。
- Python：34/34 通过，含 15 项模型包、4 项静态文本图检查。
- 独立 GCC + 固定 glslang 构建成功；软件 Vulkan 16 项通过、3 项 BF16 因驱动不支持跳过。
- 安装后的帮助入口和系统库解析通过；完整模型检查记录单独保存。
- GitHub Actions 工作流只在本地准备和验证；没有本轮远程执行结果。

这些构建测试使用小型算子或模型包占位文件，不下载真实权重，不替代上方全模型比较。

## 使用与边界

构建、转换、安装和运行命令见 [README](../../../README.md)、[运行说明](../../../docs/RUNNING.md) 和 [参考复现](../../../docs/REPRODUCE-PIPELINE.md)。本机推荐模型路径为 `models/turbo1024-s64-portable`；native PNG 和大型 trace 位于 `outputs/<results.json 中的 run>/`，未进入 Git。

默认使用 Vulkan FP16 DiT、FP32 残差/Euler 和 CPU 直接卷积 VAE。FP32 可以降低本轮误差，但不能保证通过每个提示词的严格门限，且需要更多显存。模型限制为所选静态尺寸与 32/64-token 桶。PE、原版非 Turbo CFG 路径、量化、任意尺寸、Windows/其他 GPU 和可分发二进制尚未验收。

后续优先级：独立提示词/种子数据集和跨步误差定位；FP32 attention 分块；减少逐步重复权重准备/上传；再分别评估 PE 原生 KV cache、量化和跨平台。联合 DiT 的隐藏状态每步变化，不能将跨步 K/V 复用作为精确优化。

## 来源

固定版本与运行环境见 environment/ 和每个参考 fixture。manifest.json 的 git_head 为生成本证据前的代码提交，source_files_sha256 固定实际文件。早期验证脚本的准确版本保存在 validator-source-variants/；后续运行保存 scripts/ 快照。未改写早期 [pipeline](../pipeline/README.md)、[组件](../components/README.md) 或 [单块](../dit-block/README.md) 报告，也未放宽其失败门限。

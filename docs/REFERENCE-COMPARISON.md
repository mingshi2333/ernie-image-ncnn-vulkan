# 示例项目是否也有这些问题

核对日期：2026-09-06。Z-Image 为 `c1938eef9cd7e03cb216da4f4c89a0c041543e7a`，已有 ERNIE 移植为 `8dcd6e4411137d8abe92c9d78581c4c96d5182c6`。本次核对公开源码、README、CI 和发布记录，没有运行示例模型，不能据此报告同机精度或性能胜负。

**示例确实通过不同的工程策略避开了部分问题；对于这里的严格 FP32 数值失败，尚无同条件运行证据能断言示例有或没有。**

| 问题 | 查到的示例行为 | 对本项目的含义 |
|---|---|---|
| FP16 残差超过 65504 | Z-Image 默认禁用 FP16 storage/arithmetic，启用 BF16；旧 ERNIE 的默认也是 BF16 | 这是我们引入 FP16 运行路径后必须解决的问题，不能说示例默认也会发生相同溢出。BF16 的指数范围更大 |
| 全 FP32 attention 工作区过大 | 两个示例默认低精度路径，不等于本轮全 FP32 诊断配置 | 必须固定精度、shape、设备、ncnn 版本后比较；默认 BF16 能运行不证明全 FP32 路径无问题 |
| VAE 和权重内存 | Z-Image 按 GPU heap budget 开启 host-memory weights，并计算 VAE tile 尺寸；旧 ERNIE 提供 low-VRAM 选项 | 示例有成熟的内存策略值得吸收。我们的 FP32 查询分块和 CPU VAE 直接卷积解决的是当前实现中的具体瓶颈 |
| 官方数值一致性 | Z-Image 的公开 CI 构建 Windows/Linux/macOS；旧 ERNIE README 展示 BF16 成图与官方演示图，代码有 latent/step/DiT dump 和 selftest | 本次检查的文件中未找到与本项目相同的“同初始 latent、完整 8 步、25 项固定门限”报告。不能将未公开报告解释为作者没有测试，更不能宣称本项目质量更好 |

精度和内存选项的依据：[Z-Image pipeline](https://github.com/nihui/zimage-ncnn-vulkan/blob/c1938eef9cd7e03cb216da4f4c89a0c041543e7a/src/zimage_pipeline.cpp#L88)、[旧 ERNIE runtime options](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.cpp#L293)、[旧 ERNIE CLI 默认值](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/main.cpp#L48)。验证范围的依据：[Z-Image CI](https://github.com/nihui/zimage-ncnn-vulkan/blob/c1938eef9cd7e03cb216da4f4c89a0c041543e7a/.github/workflows/CI.yml)、[旧 ERNIE README](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/README.md)。

Z-Image 的 VAE 分块还有一个实际取舍：`process_tiled` 先缩小 latent，估计 attention 和 GroupNorm 统计，再在 tiles 中复用。由这个实现可以判断，分块路径不能当作完整尺寸 VAE 逐位等价的承诺；这是源码分析，本轮没有量化其图片误差。[VAE 分块源码](https://github.com/nihui/zimage-ncnn-vulkan/blob/c1938eef9cd7e03cb216da4f4c89a0c041543e7a/src/zimage.cpp#L1042)

示例也有过运行问题：Z-Image 的 20260213 发布记录修复了 VAE 非整齐分块的越界写入，并调整 Windows 的内存使用；20260215 继续限制最大图像尺寸、降低 VAE 显存。这能证明它持续处理过此类工程问题，不能证明当前版本仍有这些已修复的 bug。[发布记录](https://github.com/nihui/zimage-ncnn-vulkan/releases)

对实施路线的影响：继续保留当前固定数值门限和负结果；同时把 BF16、长提示词、VAE 分块、低显存行为与实际生成质量作为明确的产品验证项。不能为了追求单个 FP32 样本的最后几位一致，忽略示例已经完成的实用能力。与旧 ERNIE 的公平横比仍需相同权重、初始输入、精度和硬件的实际运行。

本项目当前数据见 [注意力改进报告](../artifacts/2026-09-06/attention-parity/README.md)。网页源码通过固定 revision 下载到本地审查，文件散列和完整 Git tree 列表保存在该报告的 `evidence/outputs/reference-project-audit-v1/`；第三方源码没有并入本项目运行时。

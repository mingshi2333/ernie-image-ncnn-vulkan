# 实施路线

首版目标：官方 ERNIE-Image-Turbo 权重 → 可追溯转换 → C++ / ncnn / Vulkan → 本地 PNG。Linux、batch=1、Turbo 8 steps、CFG=1、PE 关闭。

**2026-09-05：本机已完成 1024×1024 原生生成；64×64 完整官方模块数值对照通过。** 1024 尚未完成全去噪官方参考对照或多提示词质量评估，当前是可运行的实验原型。证据见 [pipeline 报告](../artifacts/2026-09-05/pipeline/README.md)。

## 0. 上游验证与项目基础

- [x] 独立本地 Git 项目、GitHub 私有 README 占位、上游版本锁定。
- [x] 核实 ncnn 专用 KV cache allocator、容量增长和 CPU attention 更新。
- [x] CPU/Vulkan 24 个合成 GQA 场景，保存原始输出与误差。
- [x] 分开 PE 自回归缓存、Vulkan pipeline 编译缓存与 DiT 跨步近似缓存。
- [x] 64×64 分阶段执行的官方模块参考，保存 prompt、token IDs、embeddings、初始 latent、每步预测、最终 latent 和图片。
- [ ] 完整 1024 官方去噪参考及多提示词质量数据集。
- [ ] 历史项目相同输入的实际构建和复现；目前仅源码调查，不宣称已有速度优势。

## 1. 真实权重与转换契约

- [x] 官方组件选择性下载：固定 revision、Range、原始张量和组合文件散列；支持文本与 VAE 单文件权重来源。
- [x] DiT 24 / 288 / 4160 静态 token 桶，三轴 RoPE、erf GELU、GQA/SDPA、shared AdaLN 与 final norm 数学检查。
- [x] 36 个独立块：CPU FP32 / Vulkan FP32 / FP16 各 36/36，BF16 33/36；保留第 31、33、35 号失败。
- [x] BF16 磁盘流无损转换、完整计算图指纹和还原 FP32 散列检查。
- [x] 真实文本触发的第 14 号 DiT block 残差溢出复现，FP32 skip 修复；不截断激活。
- [x] 高分辨率 VAE 两处受审查 reshape 特化及独立官方目标分辨率对照。
- [ ] 多种高分辨率文本桶、精确 allocator/进程显存测量。

历史 [单 block](../artifacts/2026-09-05/dit-block/README.md) 与 [组件](../artifacts/2026-09-05/components/README.md) 报告保留原版本，低精度修复前后的数据不可混为同一版本。

## 2. Turbo CLI 功能闭环

- [x] 原生 Tokenizers 0.22.2，48 个样本与官方 token IDs 完全一致，非法 UTF-8 拒绝。
- [ ] `ignore_merges=true` 已保留，但尚未建立该开关的真实词表差异反例。
- [x] 确认实际 Mistral 文本模型分派，前 25 层实现 `hidden_states[-2]`，无 final norm；32-token 桶。
- [x] 真实英文、中文、空文本的文本路径对照；英文额外验证 Vulkan 精度路径，CLI 默认 CPU FP32。
- [x] 输入/输出层、36 blocks、C++ 时间特征与 FP32 FlowMatch Euler 8 步闭环。
- [x] BN 反归一化使用实际 pipeline 的 `eps=1e-5`，128→32 通道、2×2 unpack。
- [x] 官方 VAE decoder + post-quant；CPU FP64 GroupNorm 归约修复大尺寸误差。
- [x] CLI 的模型路径、prompt、seed、steps、device、output、输入 latent、输入 embeddings 与 trace。
- [x] 64×64 同初始 latent 完整数值/像素验收，1024×1024、8 steps、CFG=1、PE 关闭的 PNG 功能验收。
- [ ] 更长文本和更多尺寸的独立转换桶及质量验证，当前尺寸由模型包固定。
- [ ] 可迁移独立模型包、运行时完整 manifest 检验和构建 CI。

当前数值门槛不是零误差保证。64×64 FP16 最终 latent NRMSE 0.05244、PNG MAE 1.465/255；1024 仅通过功能与资源检查。FP16/BF16 scheduler 舍入不是本原型目标，scheduler 明确保留 FP32。

## 3. 设备数据流与资源优化

- [x] text encoder 在 DiT 前释放，DiT 在 VAE 前释放；预计算 embeddings 跳过文本模型。
- [x] preprocessor → 36 streamed blocks → finalizer → Euler 的设备激活路径；仅有限值状态回传，trace 另行下载。
- [x] 调用方管理 VkCompute、allocator 和共享 pipeline cache，逐块完成后释放 Net 权重。
- [x] RoPE、mask、原始文本条件预计算并驻留；每步时间特征明确生成。
- [x] 1024 单次运行的文本、逐步 DiT、VAE/PNG、总时延、峰值 RSS、整卡采样和系统 swap 前后记录。
- [ ] 分离并减少重复的权重读取、CPU 准备与设备上传，比较受控预取和权重常驻策略。
- [ ] 进一步分离文本投影、时间条件等不随相应层变化的运算，数值对照后再计算收益。
- [ ] 降低 CPU VAE 工作区（当前全流程峰值 RSS 23.03 GiB）；比较直接卷积、分块或设备解码。
- [ ] 冷启动/热运行多次测量，精确进程/allocator 显存及 swap 归属，更多设备验证。

已在 8GB 显卡、32GB 主机上完成一次 1024 原生生成；整卡采样峰值 2605 MiB，不是本进程精确峰值。系统 swap 增加 0.96 GiB，不作无 swap 声明。1024 运行总计 522.17 秒，含 trace；不据此宣称相对历史项目或官方实现提速。

## 4. PE 与后续交付

- [ ] CPU PE 使用独立原生 cache allocator、`type=1` 缓存句柄，真实模型验证 prefill/decode/reset/独立会话。
- [ ] 可选 GPU PE，缓存驻留设备，减少每 token 提交和下载。
- [ ] CPU block quantization 与 Vulkan 精度策略分开验证；不预设量化算法和收益。
- [ ] 近似 DiT 缓存必须显式启用并通过单独质量门槛，不包装成精确 K/V 复用。
- [ ] Windows、其他 GPU、便携模型包与发布材料。

下一里程碑是 **保持已通过的数学与图像结果，降低权重调度和 VAE 内存成本，并扩大 1024 质量证据**。PE、原版多步/CFG 模型、量化与跨平台在各自可验证的阶段推进。外部推送和发布按用户已授权范围执行；当前完整实现保存在本地。

# 实施路线

首版目标：官方 ERNIE-Image-Turbo 权重 → 可追溯转换 → C++ / ncnn / Vulkan → 本地 PNG。Linux、batch=1、Turbo 8 steps、CFG=1、PE 关闭。

**2026-09-06：原生生成、独立模型包、完整性检查、安装和 1024 官方对照已完成，FP32 注意力补偿与查询分块已落地。** 最新长英文 FP32 张量 24/25、PNG 通过；中文补偿版本张量 21/25、PNG 最大差仍超限。苹果历史对照通过全部门限；不能将已完成的功能等同于广泛数值/质量验收。最新证据见 [注意力改进报告](../artifacts/2026-09-06/attention-parity/README.md)，历史 [Turbo 交付报告](../artifacts/2026-09-05/turbo-delivery/README.md) 保留。

## 0. 上游验证与项目基础

- [x] 独立本地 Git 项目、GitHub 私有 README 占位、上游版本锁定。
- [x] 核实 ncnn 专用 KV cache allocator、容量增长和 CPU attention 更新。
- [x] CPU/Vulkan 24 个合成 GQA 场景，保存原始输出与误差。
- [x] 分开 PE 自回归缓存、Vulkan pipeline 编译缓存与 DiT 跨步近似缓存。
- [x] 64×64 分阶段执行的官方模块参考，保存 prompt、token IDs、embeddings、初始 latent、每步预测、最终 latent 和图片。
- [x] 完整 1024 官方去噪参考；单块 CUDA FP32 分阶段执行，禁用 TF32，保存全轨迹。
- [ ] 独立多提示词、多种子感知质量数据集与严格数值验收；不能由少量样本替代。
- [ ] 历史项目相同输入的实际构建和复现；目前仅源码调查，不宣称已有速度优势。

## 1. 真实权重与转换契约

- [x] 官方组件选择性下载：固定 revision、Range、原始张量和组合文件散列；支持文本与 VAE 单文件权重来源。
- [x] DiT 24 / 288 / 4160 静态 token 桶，三轴 RoPE、erf GELU、GQA/SDPA、shared AdaLN 与 final norm 数学检查。
- [x] 36 个独立块：CPU FP32 / Vulkan FP32 / FP16 各 36/36，BF16 33/36；保留第 31、33、35 号失败。
- [x] BF16 磁盘流无损转换、完整计算图指纹和还原 FP32 散列检查。
- [x] 真实文本触发的第 14 号 DiT block 残差溢出复现，FP32 skip 修复；不截断激活。
- [x] 高分辨率 VAE 两处受审查 reshape 特化及独立官方目标分辨率对照。
- [x] 独立导出 32/64-token 文本桶，通过完整图指纹约束复用 25 层无损权重。
- [ ] 更大文本桶、更多空间尺寸与精确 allocator/进程显存测量。

历史 [单 block](../artifacts/2026-09-05/dit-block/README.md) 与 [组件](../artifacts/2026-09-05/components/README.md) 报告保留原版本，低精度修复前后的数据不可混为同一版本。

## 2. Turbo CLI 功能闭环

- [x] 原生 Tokenizers 0.22.2，48 个样本与官方 token IDs 完全一致，非法 UTF-8 拒绝。
- [ ] `ignore_merges=true` 已保留，但尚未建立该开关的真实词表差异反例。
- [x] 确认实际 Mistral 文本模型分派，前 25 层实现 `hidden_states[-2]`，无 final norm；32/64-token 桶。
- [x] 真实英文、中文、空文本的文本路径对照；英文额外验证 Vulkan 精度路径，CLI 默认 CPU FP32。
- [x] 输入/输出层、36 blocks、C++ 时间特征与 FP32 FlowMatch Euler 8 步闭环。
- [x] BN 反归一化使用实际 pipeline 的 `eps=1e-5`，128→32 通道、2×2 unpack。
- [x] 官方 VAE decoder + post-quant；CPU FP64 GroupNorm 归约修复大尺寸误差。
- [x] CLI 的模型路径、prompt、seed、steps、device、output、输入 latent、输入 embeddings 与 trace。
- [x] 64×64 和 1024×1024 苹果同初始 latent 完整数值/像素验收，8 steps、CFG=1、PE 关闭。
- [x] 长英文 40-token 完整轨迹和单步 teacher-forcing 定位，保留 FP16/FP32 张量失败。
- [x] 中文 32-token 1024 完整轨迹及 FP16/FP32 对照，保留两种精度的张量/最大像素误差失败；不放宽门限。
- [x] 完整轨迹 → 单步 → head/block → 真实 Q/K/V → FP64 算子参考的诊断工具；定位并修正两处注意力长累加误差。
- [ ] 剩余跨步误差、文本条件与三角函数舍入的进一步隔离，独立提示词/种子回归；当前尺寸由模型包固定。
- [x] 可迁移独立模型包、全部 136 文件原生完整性检验、安装入口和损坏/搬移回归。
- [x] Linux CPU/Vulkan 构建工作流与本地 GCC + 固定 glslang 构建。
- [ ] 首次远程 GitHub Actions 执行；本轮未推送，不能把本地测试记为远程 CI 通过。

当前数值门槛不是零误差保证。1024 苹果 FP16 最终 latent NRMSE 0.01112、PNG MAE 0.10760/255；最新长英文 FP32 最终 latent NRMSE 0.0002623、PNG MAE 0.002677/255，但最后一步预测最大误差 0.11422 仍超过 0.06848。中文补偿版本最终 latent NRMSE 0.001056，部分晚期预测和解码最大误差仍超限。单步第七次预测在官方输入上通过，并不使自由运行的失败消失。FP16/BF16 scheduler 舍入不是本原型目标，scheduler 明确保留 FP32。

## 3. 设备数据流与资源优化

- [x] text encoder 在 DiT 前释放，DiT 在 VAE 前释放；预计算 embeddings 跳过文本模型。
- [x] preprocessor → 36 streamed blocks → finalizer → Euler 的设备激活路径；仅有限值状态回传，trace 另行下载。
- [x] 调用方管理 VkCompute、allocator 和共享 pipeline cache，逐块完成后释放 Net 权重。
- [x] RoPE、mask、原始文本条件预计算并驻留；每步时间特征明确生成。
- [x] 1024 单次运行的文本、逐步 DiT、VAE/PNG、总时延、峰值 RSS、整卡采样和系统 swap 前后记录。
- [ ] 分离并减少重复的权重读取、CPU 准备与设备上传，比较受控预取和权重常驻策略。
- [ ] 进一步分离文本投影、时间条件等不随相应层变化的运算，数值对照后再计算收益。
- [x] 直接卷积降低 CPU VAE 工作区，完整 1024 苹果运行峰值 RSS 由 23.03 降至 5.82 GiB，独立 VAE 对照通过。
- [x] FP32 attention 两处 Kahan 补偿与 128 查询行分块；合成 FP64 门限、受限工作区、真实 4160-token 完整矩阵逐位一致和长英文八步运行验证。中文完整 25 项轨迹与 PNG 也与补偿未分块版逐位一致，原有数值失败保留。
- [ ] 进一步减少分块同步；分块/设备 VAE 单独验证。
- [ ] 冷启动/热运行多次测量，精确进程/allocator 显存及 swap 归属，更多设备验证。

已在 8GB 显卡、32GB 主机上完成多次 1024 原生生成。直接卷积苹果运行总计 617.18 秒，包括完整包校验和 trace；长英文 FP16 为 641.17 秒、FP32 为 704.74 秒。这些是不同输入/精度下的单次记录，不是受控性能结论。历史 FP32 单步诊断整卡 100 ms 采样峰值 7609 MiB。最新分块 FP32 长英文完整运行 728.72 秒、进程 RSS 5.70 GiB，整卡 200 ms 采样峰值 4098 MiB。输入、计时范围和采样不同，不能直接计算受控速度/内存百分比收益；整卡采样包含其他进程。早期 SGEMM 运行的 522.17 秒、23.03 GiB RSS、2605 MiB 整卡显存和系统 swap 增量保留在历史报告。

## 4. PE 与后续交付

- [ ] BF16 原生 CLI 全流程与示例同条件比较；保留此前单 block 失败，不能只凭 BF16 范围更大认定质量通过。示例的内存/精度取舍见 [源码核对](REFERENCE-COMPARISON.md)。
- [ ] CPU PE 使用独立原生 cache allocator、`type=1` 缓存句柄，真实模型验证 prefill/decode/reset/独立会话。
- [ ] 可选 GPU PE，缓存驻留设备，减少每 token 提交和下载。
- [ ] CPU block quantization 与 Vulkan 精度策略分开验证；不预设量化算法和收益。
- [ ] 近似 DiT 缓存必须显式启用并通过单独质量门槛，不包装成精确 K/V 复用。
- [ ] Windows、其他 GPU、可分发二进制与发布材料。独立模型包已经完成，跨平台二进制尚未验收。

下一里程碑是 **保留已有通过与失败证据，扩大 1024 质量数据集，隔离跨步误差来源，降低权重调度和分块 attention 的同步成本**。PE、原版多步/CFG 模型、量化与跨平台在各自可验证的阶段推进。外部推送和发布按用户已授权范围执行；当前实现与模型保存在本地。

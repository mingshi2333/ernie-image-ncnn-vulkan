# 实施路线

首版交付：从官方 ERNIE-Image-Turbo 权重进行可复现转换，使用 C++ / ncnn / Vulkan 在本地生成 PNG，提供数值对照和低显存执行数据。默认 Linux、batch=1、Turbo 8 steps、CFG=1、PE 关闭。原版 50 steps/CFG、PE、其他平台和近似缓存随后独立扩展。

## 0. 上游验证与项目基础

状态：已完成原生缓存探针和单 block 参考环境，完整 pipeline 待验证。

- [x] 建立独立 Git 项目，锁定当前 ncnn 和历史参考版本。
- [x] 核实新版专用 allocator、容量增长和 CPU attention 更新。
- [x] 24 个合成算子场景检查，保存 CPU/Vulkan 原始输出和误差。
- [x] 区分 PE 自回归缓存和 DiT 跨步近似缓存。
- [x] 固定单 block 验证环境，导入 Transformers 5.2.0 的 Mistral3。版本和实际验证范围已写入锁文件，完整文本模型兼容性仍待验证。
- [ ] 官方 pipeline 关闭 PE 跑通，保留版本、prompt、token IDs、embeddings、初始 latent、sigma、逐步预测、最终 latent 和图片。
- [ ] 对旧模型做实际构建和相同输入复现，记录其成功或失败项。

退出条件：至少一套可重跑的官方小尺寸参考与完整版本记录。当前合成 attention 结果不能代替本门槛。

## 1. 一个真实 DiT block，预计 3–7 个工作日

状态：第 0 个真实权重 block 的转换与数值门槛已通过。输入为合成激活，完整真实激活和精确显存追踪仍待补齐。

- [x] 提取该 block 的 11 个官方 BF16 张量，保存版本、Range、张量和组合文件散列。
- [x] 显式导出 24、288、4160 三个 token 桶，12 组 CPU/Vulkan 比较通过，每组重复三次。
- [x] 保留原生 SDPA、4 个 RMSNorm、GELU 门控和 shared AdaLN，添加 erf 形式 GPU GELU。该 DiT 为 32/32 heads，文本侧 GQA 仍由独立缓存探针验证。
- [x] 验证 ERNIE 的三轴 RoPE，包含文本位置超过 256 的输入，拒绝不等价的标准 RotaryEmbed 自动融合。
- [x] 在运行 ncnn 前保存初始数值门槛，完成 CPU FP32、Vulkan FP32/BF16/FP16 对照。门槛适用于单 block 合成激活，完整 pipeline 质量门槛尚待建立。
- [x] 测量 4096 图片 + 64 文本 tokens，观察到 FP16/BF16 的 Flash Attention 与协作矩阵分支，记录逐次时延、进程 RSS 和设备显存采样。
- [x] 添加无损 BF16 文件存储，约 832MiB → 416MiB，还原散列和 CPU/Vulkan 输出均一致。
- [ ] 完成多个高分辨率文本长度组合及精确 allocator/进程显存峰值。当前采样是整卡总量，不能视为精确本进程峰值。

进展证据见 [复现说明](REPRODUCE-BLOCK.md) 与 [单 block 报告](../artifacts/2026-09-05/dit-block/README.md)。核心转换门槛已通过，可以进入阶段 2。8GB 方案倾向组件分阶段释放、受控权重调度和 FP16/BF16 Flash Attention。仅一个 block 的成功不能证明 36 个 block 的总权重和去噪循环可运行。

## 2. 转换闭环与 Turbo CLI，预计 1–3 周

状态：原生 tokenizer、全部 DiT blocks、输入/输出层及 FP32 latent 运算已有实现。小尺寸预测与各项负面结果以最新组件报告为准，完整文生图仍未接通。

- [x] 原生 Tokenizers 0.22.2 与官方 48 个样本 token IDs 完全一致，覆盖多语种、空白、特殊符号、BOS、空文本和 2048 截断。非法 UTF-8 明确拒绝。
- [ ] `ignore_merges=true` 配置已保留；尚未找到开关前后不同的真实词表反例，不宣称已经建立差异测试覆盖。
- [ ] 文本模型只导出实际需要的文本路径，核实 `hidden_states[-2]` 对应截断层及 final norm 行为。
- [x] 转换 36 个 DiT blocks、preprocessor、finalizer，保留来源与 checksum。独立 block 检查中 CPU FP32 / Vulkan FP32 / FP16 均为 36/36，BF16 为 33/36，失败项不作通过处理。
- [x] 实现 FP32 FlowMatch Euler：shift 4、`t = 1000 * sigma`，21 组 CPU/Vulkan 合成预测测试通过，sigma、时间步和每步更新逐位对齐。
- [ ] 将真实 DiT 预测接入 8 步循环，补齐 C++ 时间正弦特征生成与 FP16/BF16 scheduler 的舍入顺序。
- [x] 实现 VAE 前的 BN 反归一化（实际 pipeline 使用 `eps=1e-5`）及 128→32 通道、2×2 unpack，验证非方形布局。
- [ ] 转换并验证实际 VAE decoder。
- [ ] 实现 model path、prompt、seed、resolution、steps、device、output、输入 latent 和输入 embeddings。
- [ ] 小尺寸先通过，再验收 1024×1024、8 steps、CFG=1、PE 关闭的 PNG 输出。

退出条件：官方权重 → 转换 → C++ CLI → PNG 可重跑，逐模块和每步误差可追溯。相同 seed 不足以保证跨框架同初始噪声，比较时必须读取同一份 latent 文件。

## 3. 设备数据流与 8GB 内存方案，预计 1–3 周

- [ ] text encoder 在 DiT 加载前释放，DiT 在 VAE 加载前释放。预计算 embeddings 模式不加载文本模型。
- [x] 接通 preprocessor → streamed DiT → image-token slice → finalizer 的 `VkMat` 路径，组件之间不下载激活；Euler 闭环尚待接入。
- [x] 调用方管理 `VkCompute`、allocator 和提交边界，拒绝计算图中没有 Vulkan 实现的计算层；逐块完成后释放该 Net 的权重。
- [ ] 将 RoPE、mask、原始文本投影和每步时间嵌入按依赖关系预计算，保持特定层的广播而不展开为大张量。
- [ ] 比较 host-memory weights 与受控 block 权重调度，记录 PCIe/host 访问成本。必要时 VAE tiled decode。
- [x] 实现调用方共享 Vulkan pipeline cache，复用相同静态图的 pipeline；同输入对照结果见组件报告。缓存权重、优化上传和重复去噪仍需单独测量。
- [ ] 输出冷启动、热运行、文本编码、逐步 DiT、VAE、总时间、峰值 RAM/VRAM、swap 增量和模型大小。

退出条件：在本机明确报告支持的最大分辨率/文本长度/精度与资源峰值。8GB 是待验证目标，不把部分算子成功或单独低显存选项视作达成。

## 4. PE 与交付完善

- [ ] CPU PE 接专用 cache allocator，CPU cache 使用 `type=1`，按当前 x86 attention 做匹配基准。
- [ ] 在实际模型上验证每层 cache 生命周期、prefill、decode、重置和并行独立会话。
- [ ] 可选 GPU PE：缓存始终留在设备，尽量将 decoder / LM head 放入合理提交批次，仅下载生成所需的小输出。
- [ ] CPU block quantization、Vulkan 精度策略分别量化评估。当前 ncnn 不支持 Vulkan Gemm block-quantized weights，不预设量化算法或收益。
- [ ] 添加构建 CI、模型 manifest 校验、可重复基准和错误处理，再扩展 Windows 与其他 GPU。

退出条件：PE 开启/关闭分开报告，不能将二者作为同一配置比较速度。发布前所有公开兼容性和性能声明都由实测支撑。

## 决策顺序

下一个里程碑是 **真实文本条件下的小尺寸 Turbo 闭环**：验证 Mistral3 文本路径和 hidden-state 选择，接入 C++ 时间特征与 8 步 Euler，再完成 VAE 解码。比较时使用同一组保存的 embeddings 和初始 latent。先以 FP16 作为低显存工作路线，BF16 的独立失败门槛保持完整可追溯。

不承诺整模型提速倍数。当前已实现分阶段加载约 14.9GiB DiT 权重的基础路径，加载/初始化开销、真实文本条件下的误差积累和 8 步闭环仍需验证。原先 4–8 周是单人工程投入的粗略估算，单 block 或一次预测的耗时不能用来预测整项目进度或端到端速度。

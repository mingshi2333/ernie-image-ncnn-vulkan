# 实施路线

首版交付：从官方 ERNIE-Image-Turbo 权重进行可复现转换，使用 C++ / ncnn / Vulkan 在本地生成 PNG，提供数值对照和低显存执行数据。默认 Linux、batch=1、Turbo 8 steps、CFG=1、PE 关闭。原版 50 steps/CFG、PE、其他平台和近似缓存随后独立扩展。

## 0. 上游验证与项目基础

状态：已完成原生缓存探针，完整参考环境待验证。

- [x] 建立独立 Git 项目，锁定当前 ncnn 和历史参考版本。
- [x] 核实新版专用 allocator、容量增长和 CPU attention 更新。
- [x] 24 个合成算子场景检查，保存 CPU/Vulkan 原始输出和误差。
- [x] 区分 PE 自回归缓存和 DiT 跨步近似缓存。
- [ ] 固定兼容的 PyTorch / Diffusers / Transformers 版本。当前锁中 Transformers revision 仍待实际参考环境验证，不能随意把最新版本标为兼容。
- [ ] 官方 pipeline 关闭 PE 跑通，保留版本、prompt、token IDs、embeddings、初始 latent、sigma、逐步预测、最终 latent 和图片。
- [ ] 对旧模型做实际构建和相同输入复现，记录其成功或失败项。

退出条件：至少一套可重跑的官方小尺寸参考与完整版本记录。当前合成 attention 结果不能代替本门槛。

## 1. 一个真实 DiT block，预计 3–7 个工作日

- [ ] 确认权重容量和分片位置，只加载该 block 及必要前后处理，避免全模型 FP32 导出。
- [ ] 显式导出单 block。固定 shape 起步，再覆盖少量 token 长度桶。
- [ ] 导出中保留 SDPA、GQA、RMSNorm、GELU 门控和 shared AdaLN 的语义。
- [ ] 单独验证 ERNIE 的三轴 RoPE（theta 256，轴维度 32/48/48，完整重复角度和非交错半向量旋转），不能照搬 Z-Image 或标准 RotaryEmbed。
- [ ] 在 CPU FP32、Vulkan FP32/BF16/FP16 下比较同一输入。依据 PyTorch 自身后端差异，在查看候选结果前写下真实 block 的精度阈值。
- [ ] 测量 1024 输出对应的 4096 图片 tokens 加不同文本长度，确认 Flash Attention 实际分支与显存峰值。

退出条件：真实权重 block 能从原始权重重建并通过数值门槛，记录 CPU/GPU 传输和时延。此时更新完整实施估期与 8GB 容量方案。

## 2. 转换闭环与 Turbo CLI，预计 1–3 周

- [ ] Tokenizer 按官方配置处理 Unicode、ByteLevel、`ignore_merges`、BOS、截断和 special tokens。固定中英日、空白、符号、长文本样本，token IDs 完全一致。
- [ ] 文本模型只导出实际需要的文本路径，核实 `hidden_states[-2]` 对应截断层及 final norm 行为。
- [ ] 导出全部 DiT blocks、preprocessor、finalizer，保留每一文件的来源、checksum 和转换命令。
- [ ] 实现 FlowMatch Euler：固定 shift 4，`t = 1000 * sigma`，严格匹配时间步序列与更新顺序。
- [ ] VAE decoder 对齐 128 通道 packed latent、2×2 unpack、32 通道 latent 和 BN 反归一化。以锁定官方 pipeline 实际使用的 epsilon 为准。
- [ ] 实现 model path、prompt、seed、resolution、steps、device、output、输入 latent 和输入 embeddings。
- [ ] 小尺寸先通过，再验收 1024×1024、8 steps、CFG=1、PE 关闭的 PNG 输出。

退出条件：官方权重 → 转换 → C++ CLI → PNG 可重跑，逐模块和每步误差可追溯。相同 seed 不足以保证跨框架同初始噪声，比较时必须读取同一份 latent 文件。

## 3. 设备数据流与 8GB 内存方案，预计 1–3 周

- [ ] text encoder 在 DiT 加载前释放，DiT 在 VAE 加载前释放。预计算 embeddings 模式不加载文本模型。
- [ ] 连接 preprocessor → DiT → image-token slice → finalizer → Euler 的 `VkMat` 路径。
- [ ] 调用方管理 `VkCompute` 和提交边界，检查是否有层回退 CPU 或隐式下载。
- [ ] 将 RoPE、mask、原始文本投影和每步时间嵌入按依赖关系预计算，保持特定层的广播而不展开为大张量。
- [ ] 比较 host-memory weights 与受控 block 权重调度，记录 PCIe/host 访问成本。必要时 VAE tiled decode。
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

最近的首要里程碑是 **一个真实权重 DiT block 的转换和 CPU/Vulkan 对照**。合成 KV cache 探针已经验证了新版接口，继续只优化 PE 会偏离 PE 关闭时的主要文生图瓶颈。

阶段 1 结束前不承诺整模型提速倍数。各阶段可能重叠，已有移植可复用程度仍需实际权重复现决定。当前按熟悉 C++/ncnn 的单人投入估算可靠首版约 4–8 周，8GB 优化和跨平台结果可能继续增加工作量。

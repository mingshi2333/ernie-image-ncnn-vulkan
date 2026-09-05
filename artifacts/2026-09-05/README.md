# Native KV cache probe — 2026-09-05

状态：24 个场景检查通过。验证对象为合成 causal GQA attention，未使用 ERNIE 模型权重，也未测试完整文生图速度。

硬件：AMD Ryzen 7 7745HX，NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 显存，驱动 595.91.07。ncnn `6a1bf000f363714839a36793addc8c879d3d899e`，Clang，Release，4 CPU threads，系统 glslang 16.2.0。完整构建设置和文件 checksum 由 `verified-run/manifest.json` 记录。

## 方法

- Q heads=32，KV heads=8，head dimension=128，batch=1。
- 输入为确定性伪随机值。独立参考使用 double 累加，按因果可见范围直接计算 GQA 和 stable softmax，不读取 ncnn 的缓存。
- 场景 A：32 token prefill + 8 次单 token decode，容量提示 128，检查预留空间复用。
- 场景 B：重置后依次追加 5/1/1/1/9/17/31 tokens，总计 65，容量提示 8，检查超过提示值时正确扩容。
- 场景 C：再次重置，换一条输入序列，7 token prefill + 2 次 decode，检查旧历史不影响输出。
- CPU FP32 两种 allocator 设置，Vulkan FP32 / FP16 / BF16 各两种 allocator 设置，合计 8 个配置 × 3 场景。
- Vulkan FP16/BF16 使用低精度存储，`use_fp16_arithmetic=false`。这不是整模型混合精度策略的验证。

## 结果

| 配置 | 场景 A：K/V 各出现的底层缓冲数，默认 → 专用 allocator | 三个场景的最大绝对误差 | 检查 |
|---|---|---|---|
| CPU FP32 | 1 → 1 | 2.161e-7 | 6/6 通过 |
| Vulkan FP32 | 9 → 1 | 1.788e-7 | 6/6 通过 |
| Vulkan FP16 storage | 9 → 1 | 5.841e-4 | 6/6 通过 |
| Vulkan BF16 storage | 9 → 1 | 4.539e-3 | 6/6 通过 |

误差门槛在运行前设定：FP32 最大绝对误差与 NRMSE 均 ≤2e-5；FP16 分别 ≤0.002 / 0.005；BF16 分别 ≤0.015 / 0.03。NRMSE 为 `sqrt(sum(error²) / sum(reference²))`。这些门槛仅用于输入在约 [-1,1] 的小型算子检查，不能沿用为 ERNIE block 或最终图片验收标准。

底层缓冲数通过缓存 handle 的 backing identity 变化观测，包含第一次分配，不是所有底层 allocator 调用总数，也不是峰值显存。新增有效 token 仍需写入缓存，未测量并宣称“完全零拷贝”。CPU 默认路径已经能复用容量，而当前 Vulkan 要使用独立 cache allocator 才能取得此项复用收益。

原始首轮输出：[CPU](cpu.jsonl)、[Vulkan](vulkan.jsonl)。加入可重跑记录工具后的输出与构建来源见 [verified-run](verified-run/manifest.json)。

## 尚未验证

- 旧 ncnn 和新 ncnn 的匹配时延 A/B。
- PE 完整 26 层的 token 一致性与吞吐。
- 多会话交错/并发、beam 分支复制和跨网络 cache ABI。
- DiT 的 4096+文本 tokens 长序列、36 层、8 步误差和实际 Flash Attention shader 分支。
- 整模型 BF16/FP16、量化、峰值 RAM/VRAM、1024×1024 成图或端到端提速。

下一步按路线验证真实 DiT block，这份探针作为缓存接口回归保留。

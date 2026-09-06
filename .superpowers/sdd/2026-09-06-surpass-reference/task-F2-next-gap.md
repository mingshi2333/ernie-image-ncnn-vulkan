# F2 next-gap decision

现有公开development证据已覆盖固定1024×1024的strength0真实跳过路径和strength0.5绝对步4..7。下一最小缺口是同一输入的strength1端点：它首次验证initial逐字节等于saved noise并执行原schedule绝对步0..7。它不引入新shape、权重或模型数学。

新增准备合同固定完整分母：official suffix 25张量，端到端29张量加PNG；任何missing、duplicate、unexpected、非finite或错误绝对步均不得通过。真实GPU执行仍需独立准备复核和根任务串行调度。

暂不先扩shape，因为每个新shape都需新的官方encoder fixture、shape-only图证据、原生三边界、trusted registry和schema-3包认证。暂不进入formal15，因为adapter还必须证明两侧实际消费冻结image/decoded RGB/saved noise/strength并逐case绑定shape/model/stage precision/PE；早期缺配置calibration也不是baseline。本切片没有读取formal prompts或运行formal inputs。

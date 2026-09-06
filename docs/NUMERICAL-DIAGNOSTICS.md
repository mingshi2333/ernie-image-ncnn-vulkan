# 数值误差与显存诊断

完整生成的数值回归继续记录 `tools/validate_pipeline.py` 的固定门限。必须复用保存的初始 latent，保留失败张量、最大像素差和运行失败。单个算子更精确、单步通过、平均像素误差较小，都不能代替完整轨迹的数值对照。

## 门槛的来源与适用范围

2026-09-07 补充：FP32 的 `pixel_max ≤ 2`、`pixel_mae ≤ 0.1`、`NRMSE ≤ 0.003` 及相关张量误差门槛，均由助手在本项目中选择。它们已出现在 2026-09-05 的首版验证器提交 `39678e8`；原注释明确限定像素门槛用于小型夹具诊断，不构成感知画质声明。它们不是从 ERNIE、ncnn 或 futz12 采用的标准，也不是用户逐项指定的数值。低精度、文本条件和后续“超过”计划中的收益目标同样属于本项目制定的工程协议。

这些数值尚未通过官方不同计算后端的正常波动、代表性分辨率/提示词和人工画质评价完成标定。事先固定门槛有助于防止事后按结果改规则，但不能证明门槛适用于全部尺寸，或能直接判断画面质量。此前将严格数值回归状态泛称为“质量失败”，容易夸大结论；后续分别报告：

- **执行与正确性事实**：是否完整出图、形状/输入是否正确、是否存在 NaN/Inf、崩溃、分配失败或模型数学不一致。每项需要对应证据。
- **历史数值回归状态**：相同输入下的张量误差、像素误差分布，以及是否通过原有固定规则。历史 `quality_gate_failed` 表示未通过该协议，不能单独推出程序不可用或内存不足。
- **感知质量与实用比较**：提示遵循、文字/数量/关系、可见瑕疵、实际速度和资源使用。需要代表性图像及双方实际运行证据；像素门槛通过也不保证这些方面通过。

例如，已完成的 1376×768 原生苹果图在 3,170,304 个 RGB 通道值中，仅 4 个通道值、分布于 3 个像素的误差达到 3，平均误差为 0.005041（0–255 色阶）。它仍未通过历史 max≤2 规则，但这个统计本身不足以认定画质有实质下降。该次完整运行另有两项张量最大误差超限，全部张量有限且原生程序返回 0，详见[保留的完整记录](../artifacts/2026-09-07/fixed1376-native-pipeline/README.md)。

后续标定使用开发输入检查官方计算后端差异、同输入的参考项目输出和感知质量；冻结的正式语料不用于调门槛。若证据支持调整，建立带理由的新协议版本，并同时保留旧协议结果。本次说明未修改验证器、验收 JSON、既有阈值或封存结果，也未将正式质量验收改为通过。

参考项目的首次实际 64×64 独立解码中，CPU/Vulkan FP32 相对保存的官方 PNG 都是 max1、MAE0.000732，均通过历史像素规则；双方 CPU/Vulkan 图片之间有两个通道相差 1。它仅提供小尺寸解码证据，尚不能标定高分辨率自由去噪轨迹的门槛。见[参考项目解码记录](../artifacts/2026-09-07/reference-decoder64/README.md)。

## FP32 注意力

固定 ncnn 的非 Flash Vulkan SDPA 在 softmax 分母与概率乘 V 两处逐项累加长向量。项目用 Kahan 补偿降低这两处 FP32 累加误差，乘法仍为 FP32。GLSL 的 `precise` 保留补偿所需的运算顺序。构建时校验两个原始 shader 的完整 SHA256，然后生成派生源码；上游 shader 改变时需要重新审查，不能跳过指纹检查。两个源码都是 CMake 的重新配置依赖，已有构建目录也会检测后续变更。

长查询按最多 128 行处理，每行仍使用全部 K/V 和对应 mask。单个注意力矩阵的存储从 `heads × queries × keys` 降到 `heads × min(128, queries) × keys`。4160 个图文 tokens、32 个头时，FP32 分数矩阵从 2,215,116,800 字节降到 68,157,440 字节，即约 2.06 GiB 降至 65 MiB。这不是整个进程或显卡的峰值；权重、Q/K/V、完整 mask、MLP、其他缓存仍占用内存。

每个分块完成后同步，以便安全复用临时存储，不下载激活。代价是更多提交：4160 个查询需要 33 个分块。`BlockSequenceStats.compute_submissions` 记录层内提交及调用方最后一次提交；36 个 DiT block 共 1188 次，不包含输入/输出 head、Euler 和诊断下载。不能继续套用旧版每个 block 一次提交的计数。

该路径适用于项目的 batch=1 FP32 三维 Q/K/V 和匹配的二维/三维 mask。原生 KV cache 的管理、追加与会话生命周期继续由 ncnn 处理；低精度 Flash 路径不变。不同去噪步之间不复用 DiT K/V。

以下测试不依赖权重：

```sh
ctest --test-dir build --output-on-failure \
  -R 'attention_long_fp32_accuracy|attention_bounded_workspace|runtime_cache'
```

- 长向量测试包含常量 V 列、带符号抵消和 mask，使用独立 FP64 oracle。
- 工作区测试以 300 KiB 限制单次临时分配，覆盖跨分块的共享/逐头 mask、无 mask、GQA、不同 Q/K 长度和 V 宽度，以及最后一个不完整分块；同时比较分块与完整修正版的 FP32 输出。
- KV cache 测试覆盖 CPU/Vulkan、默认/专用 allocator、追加、重用、重置与容量增长。CPU cache 仍以 `type=1` 提取。

`register_attention(net, false)` 仅供诊断完整矩阵路径，默认启用工作区限制。`attention_internal_submissions(net)` 只适用于已注册项目注意力实现的 Net。ncnn 的 shader 反射要求 push-constant block 命名为 `parameter`；派生层会检查复制 shader 的反射结果，并拒绝执行未成功创建的 pipeline。

## 从完整轨迹缩小到算子

1. 用相同权重、初始 latent 和保存的官方参考执行完整自由轨迹，先确定失败步骤。
2. 使用 `tools/diagnose_pipeline_step.py`，将该步骤的官方输入喂给 native。通过只表明这一步的局部误差较小。
3. 用 `tools/diagnose_dit_stages.py` 保存输入 head 和每个 DiT block 的输出，检查误差首次增大和逐层放大的位置。可分别查看图像与文本 token。
4. 用 `ernie-block-runner --output-blob NAME` 提取中间结果，结合独立的 FP64 算子参考定位舍入来源。

例如：

```sh
.venv/bin/python tools/diagnose_dit_stages.py \
  --model models/turbo1024-s64-portable \
  --reference outputs/pipeline1024-chinese-s64-fp32-v1/reference \
  --step 0 --precision fp32 --output outputs/chinese-stages-new
```

阶段诊断复用官方 latent、文本和 CPU 时间特征。native 使用保存的 CPU cos/sin 表；官方 block 在指定参考设备上计算 cos/sin，CPU/CUDA 的最后几位差异仍在比较范围内。中间输出必须显式匹配轴顺序：例如官方 Q norm 的 `B,S,H,D` 与 native 的 `H,S,D` 不能直接按扁平数组比较。

单独构造 ncnn `MemoryData` 测试模型时，带类型标签的权重流必须指定 `21=0`；默认原始流模式不能读取带标签的测试数据。早期错误夹具和错误轴比较应记录为无效诊断，不作为模型质量证据。

## 区分文本条件与去噪误差

验证器可用 `--reference-embeddings` 跳过 native 文本编码，读取同一份官方 FP32 特征。结果明确标记为 `saved_reference_diagnostic`；即使通过，也不算完整 native prompt → PNG 验收。此选项不会改变时间特征、位置表、去噪器、VAE 或门限。

CPU RMSNorm 的 FP64 归约候选在真实第一层上更精确，但完整中文文本编码对官方的 NRMSE 从 6.2544e-6 略增至 6.7566e-6，候选因此未设为默认。保留这一负结果，避免将局部精度改善直接当作全模型收益。

Q·K 打分的 Kahan 补偿候选也未采用。真实中文 step0/block15 的全部 17,039,360 个值比较中，L2 从 .198776545 降至 .193962379，但最大误差从 .006591797 升至 .006835938；未满足预先固定的两项均改善条件，因此没有进入完整出图验证，实验代码已从主实现移除。OFF 对照逐位复现旧输出，两种构建各通过 23 个注意力与缓存用例，均无内存上限或 OOM 事件。见 [实际记录](../artifacts/2026-09-07/compensated-qk-experiment/README.md)。这属于计算路径差异，不能据此将已有完整图像失败归因为内存不足，也不能确认单一数值根因。

## 独立复核与证据保存

```sh
.venv/bin/python tools/collect_parity_evidence.py \
  --run outputs/pipeline1024-chinese-s64-fp32-kahan-v1 \
  --output artifacts/parity-audit-new
```

工具核对参考、native 张量、图片、二进制、脚本和模型清单散列，确认门限未修改，并使用 NumPy 重新计算全部误差、量化和 verdict。完整执行中的数值失败可封存；没有跑完的生成需要单独保留 `result.json` 与日志，不能伪装成图像对照。`--evidence FILE` 可额外保存这类小型失败证据；`--report FILE.md` 将审阅后的说明一起纳入证据散列。

早期实验只记录验证器散列，准确版本存于 `outputs/validator-source-variants/<sha256>.py`。工具按散列校验并保存这份源文件，将覆盖范围标为 `validator_only`；后续逐运行脚本快照标为 `per_run_scripts`。现代快照缺失或损坏时必须报错，不能退回旧格式掩盖证据缺失。

每次长运行均使用独立目录和二进制快照；大型 GPU 工作依次执行。整卡显存采样包含桌面及其他进程，不等于本程序 allocator 峰值。模型检查、阶段跟踪和下载都会改变计时范围，性能结论需要另外设计受控实验。

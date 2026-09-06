# P1 数值正确性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 关闭已知完整轨迹失败，建立可以阻止性能优化引入退化的正确性基线。

**Architecture:** 用已保存张量做交叉实验，将条件误差、DiT 自由轨迹误差和 VAE 解码误差分别定位。修复进入发生问题的组件；原生完整路径仍是最终验收对象。

**Tech Stack:** 现有官方参考、NumPy/FP64 诊断、ncnn CPU/Vulkan、CTest、现有 trace/probe。

## Global Constraints

继承 [总计划](2026-09-06-surpass-reference.md) 与 [固定门槛](surpass-acceptance.json)。单步通过不能关闭自由轨迹失败；已有通过项不能因本阶段改动回退。以下新测试、命令及类型为待实现接口。

## Task Q1: PE 图像的 VAE 交叉解码

**Files:** Create `tools/diagnose_vae_cross.py`, `tests/test_vae_cross_report.py`；必要时扩展 `probes/head_runner.cpp` 的 `--component vae` 诊断输入和 `tools/validate_dit_heads.py` 的既有 VAE 验证，不变更正常路径。读取 `outputs/pipeline512x384-pe-fp32-v1` 的 official/native unpacked latent。

**Interfaces:** `cross_terms(oo, no, on, nn: np.ndarray) -> dict`；命名为 `oo=official_decoder(official_latent)`、`no=native_decoder(official_latent)`、`on=official_decoder(native_latent)`、`nn=native_decoder(native_latent)`。结果分别记录解码实现差、输入轨迹差、全路径差及交互项，所有数组必须形状一致且有限。

- [ ] 写报告测试，确保不能把输入误差全部归因给 VAE：

```python
def test_cross_terms_separate_decoder_and_input_error(self):
    z = np.zeros((3, 2, 2), dtype=np.float64)
    terms = cross_terms(z, z + .01, z + .1, z + .11)
    self.assertAlmostEqual(terms["decoder_on_official_max"], .01)
    self.assertAlmostEqual(terms["input_through_official_max"], .1)
    self.assertAlmostEqual(terms["interaction_max"], 0.0)
```

- [ ] 执行该 unittest 先得到失败，再实现四组结果的独立比较；不将误差“百分比归因”当作线性可加的因果证明。
- [ ] 依次执行四次 512×384 解码，先退出官方进程再启动 native，固定 direct convolution 与当前 GroupNorm；核对输入 bytes，保存四个输出和交叉报告。
- [ ] 若 `no-oo` 超门槛，按 post-quant、各 decoder block、GroupNorm、输出卷积分段抓取；若该项通过而 `on-oo` 主导，进入 Q2 追溯 latent。不得只因为 decoded 超限就改 VAE。
- [ ] 对定位出的算子编写最小真实激活 regression，保存小型输入散列/必要数据，修复后先跑该 regression，再重跑完整 PE 图像。
- [ ] 全部 25/25 和 PNG 通过才关闭该 fixture；否则报告新的边界并保留失败。本地提交 `fix: isolate and correct connected VAE parity drift` 仅在存在实际修复时使用；纯诊断提交使用 `test: record crossed VAE decoder and input errors`。

**Acceptance:** 确定哪一部分产生主要差异，至少有一项可复核的交叉证据；目标为 PE 完整图像 25/25，不能将诊断完成写成质量通过。

## Task Q2: 长文本自由轨迹与精度策略

**Files:** Extend `tools/diagnose_pipeline_step.py`（增加 BF16，门槛与现有低精度一致）、`tools/diagnose_dit_stages.py`、`tools/validate_pipeline.py`；Create `tools/diagnose_trajectory.py`, `tests/test_trajectory_report.py`；只按证据修改 `src/text_encoder.cpp`, `src/conditioning.cpp`, `src/ernie_attention.cpp`, `src/denoiser.cpp` 或对应数值算子。

**Interfaces:** `first_failed_boundary(rows: list[dict]) -> str | None` 返回按真实执行顺序的首次越界点，最小输入字段为 `boundary, passed`；完整报告行另外必须含 `step, stage, precision, input_sha256, reference_sha256, metrics`，由报告验证器检查。报告标明 official/native text、teacher-forced/free-running，不混合统计。

- [ ] 写出首次边界判定的测试，晚期最大误差更大也不能覆盖更早的首次失败：

```python
def test_first_failure_is_not_largest_late_error(self):
    rows = [{"boundary": "step-1/block-4", "passed": False},
            {"boundary": "step-7/final", "passed": False}]
    self.assertEqual(first_failed_boundary(rows), "step-1/block-4")
```

- [ ] 先核对历史长英文、中文和 1080-token 的 source/runner/gates；从首个误差突增前一步开始，对照官方输入与 native 输入各自的一步预测，再缩小到 head/block，必要时 Q/K、RoPE、softmax、P@V、GELU、残差/Euler。
- [ ] 分开检查真实文本有效长度与 padding、图像/文本 token 轴、RoPE CPU/CUDA 舍入、FMA/归约次序、输出层/BN；复用已有 `--reference-embeddings` 仅作诊断。增加任何算子候选前写它解释哪条观测，以及会改善哪项完整结果。
- [ ] 用独立 FP64 小算子参考判断累加候选；保持 FP32 residual、Euler、位置表和 finite 检查；拒绝 clamping、提示词 ID 特判和跨去噪步 K/V 缓存。
- [ ] 每个候选先过真实激活 probe，再跑受影响的完整 fixture，再跑已有苹果通过项。一次只改变一个数学/精度因素，正式 benchmark 不与 trace 并行。
- [ ] 形成两个明确配置：`reference=FP32`；`default=在相同质量门槛下选出的最快配置`。默认配置只能由设备能力/固定模型配置决定，不按正式样本身份切换。修复完成前 BF16 保留显式实验模式；历史 BF16 的 block 31/33/35 和连接长文本失败仍是待关闭项，不能改标签伪装完成 P1。
- [ ] 执行 `.venv/bin/python tools/validate_pipeline.py --model models/turbo512x384-s2048-portable --prompt-file outputs/futz12-comparison-v1/source/assets/prompt.txt --precision fp32 --reference outputs/pipeline512x384-prompt1080-fp32-v1/reference --output outputs/q-long-fp32-v2`；低精度同输入另建目录。对 1024 fixtures 使用各自原始匹配包和参考，不复用不同尺寸的参考。
- [ ] 相关 CTest/unittest 与完整 fixtures 通过后提交组件修复，并记录每条历史失败是否关闭。没有通过的模式继续阻止对应质量声明。

**Acceptance:** 64/1024 苹果保持通过，1024 长英文/中文、512×384 PE 和 1080-token 的 FP32 全门槛通过；已记录的长英文/中文 FP16、1080-token BF16 及独立 BF16 block 失败用最终候选在原门槛下关闭。默认性能配置通过开发集和这些历史输入的适用门槛。某种精度仍失败时 P1 对应项不关闭，允许继续独立诊断，不能用移出默认配置抵消失败。

## Task Q3: PE 与独立验收集的保护

**Files:** Extend `tools/reference_pe.py`, `tools/validate_pe.py`, `tools/validate_pe_tokenizer.py`, `tests/test_pe_sampling.cpp`, `probes/pe_block_runner.cpp`。复用 `src/prompt_enhancer.cpp` 与 `src/pe_session.cpp` 的真实入口；不在 Python 重写 sampler 后将它冒充 C++ 实测。

**Interfaces:** 原有 PE CLI 保持；新增参考/验证批次清单包含 `prompt,width,height,max_tokens,mode,temperature,top_p,input_ids_sha256`。PE greedy 所有 logits 使用固定 2e-4 门槛；随机采样只比较已知 logits 下的分布，不要求同 seed 跨实现逐 token 相同。

- [ ] 在既有 C++ 测试里调用 `ernie::sample_pe_token`，使用 logits `[log(.7), log(.2), log(.1)]`、temperature=1、top_p=1 抽样 100000 次，固定本地 seed，每项频率误差 ≤ 0.01；增加 top-p 0.6 只保留第一项，保留已有 greedy tie 检查。运行 `ctest --test-dir build -R pe_sampling_contract --output-on-failure`；EOS 与长度上限在真实 PE runner 验证。
- [ ] 选 12 条 PE 输入，包含中英日文、引号/换行、空白、接近 2048 输入和长输出；保存完整 official greedy 参考，分批独立进程执行。超限样本测试拒绝，不触发超容量推理。
- [ ] 所有 token/文字/适用 logits 通过；reset/不同会话交错与容量边界仍通过，检查缓存只含真实历史 token。
- [ ] P1 只运行开发/历史集与这 12 条 PE 输入，建立后续优化的回归基线；P0 正式 72 条的结果保留到 P5 冻结最终候选后才揭示。开发集有意包含已知问题，不能称为独立盲测。
- [ ] 对误差与 source/runner SHA256 独立复核，冻结 `artifacts/<执行日期>/correctness-v1/`；提交 `test: gate native generation and PE precision profiles`。

**Acceptance:** 数值正确性进入可重复门槛，后续优化有明确回退基准；人工盲评和正式胜出判定由 P5 执行。

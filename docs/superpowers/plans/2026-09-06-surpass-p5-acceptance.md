# P5 最终比较与判定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用冻结版本、完整分母和原始证据回答是否真正超过参考项目；结论必须由所有硬门槛共同决定。

**Architecture:** 测量工具只产生记录，验收器只读不可变输入/目标和原始结果；人工盲评使用隐藏实现名的图片，不由模型代替人工投票。数值、产品质量、速度、内存、交付和结构分别判定。

**Tech Stack:** P0 比较工具、P1 固定验证器、P2 计数器、Python unittest/NumPy、离线人工评分页面、现有 artifact 来源快照。

## Global Constraints

继承 [总计划](2026-09-06-surpass-reference.md) 和 [机器可读目标](surpass-acceptance.json)。以下新接口与命令待实现。formal results 不用于默认策略调参；一旦按正式失败作修复，该集合变为回归资料，新的独立声明另冻结集合版本，并保留第一次失败。

## Task V1: 冻结候选并执行完整质量验收

**Files:** Extend `tools/acceptance_manifest.py`, `tools/compare_ports.py`, `tools/validate_pipeline.py`, `tools/validate_img2img.py`, `tools/validate_pe.py`, `tools/source_inventory.py`；Create `tools/blind_review.py`, `tools/summarize_blind_review.py`, `tests/test_blind_review.py`。

**Interfaces:** `blind_review.py --manifest FILE --results DIR --output NEWDIR` 生成离线页面、匿名配对表与独立私有映射；`summarize_blind_review.py --votes DIR --mapping FILE --output NEWDIR` 校验投票完整性并按 prompt 聚类计算区间。评分记录包含 `reviewer_id, pair_id, prompt_following, text, count_relation, artifacts, overall_choice`；`overall_choice=left/right/tie` 用于主指标，分项用于解释。

- [ ] 先按 P0 预声明网格，在开发集校准最终候选与固定对方，选出通过共同质量标准的最快对方配置与精度匹配的候选。随后冻结 candidate commit、完整源码、二进制、依赖、模型包、精度/设备策略、语料与 gates 的 hash；候选运行时目录只读，禁止长运行中重新构建覆盖 runner。
- [ ] 再核对参考 HEAD。主结论仍对固定 `8dcd6e4`；若上游更新，另记录新版本差异和补测范围，未经新版本完整对照不能写“超过上游最新版”。
- [ ] 重跑所有历史失败/通过输入；FP32、历史 FP16/BF16 的适用门槛保持原值，旧负结果原文不动。确认 P1 修复被最终候选保留。
- [ ] 正式集 24×3=72 个 case 分别检验 FP32 与生产默认配置；两者确为同一配置时复用同一不可变运行记录，否则各跑 72 次。每条同时校验原生 token/text、25 个或协议明列的张量边界、最终 RGB/PNG；固定官方 FP32 参考分阶段执行且禁用 TF32，不能拿 ours FP32 当官方 oracle。
- [ ] 对方生成相同 72 个输入的产品结果，记录其质量门槛、输出和失败。若只能做产品级对照，保持来源/精度标签；不能用同 seed 替代相同实际噪声。双方 PE 性能样本使用 greedy，文字/token不同必须显式记录并检查可比性。
- [ ] 额外运行 12 条 PE greedy、token 容量边界、F1 十个尺寸 canary/三个极端尺寸、F2 的 15 条图生图输入/强度组合与四种模式。所宣称为生产支持的额外精度/设备组合必须有独立适用证据；未验收模式明确标实验。
- [ ] 生成 72 对匿名图片，至少两名相互独立的人类评分。先测试匿名映射可逆而页面/文件名不泄漏实现名；重复或缺失投票报错。脚本不得生成模拟投票当正式结果。
- [ ] 固定 bootstrap seed、10000 次按 24 个 prompt 聚类重采样，保留每个 prompt 的三个 seed 与所有评审票。报告 95% 区间；下界 ≥.45 才证明预设的不劣，> .5 才允许更高画质声明。评审人数、样本量不足或区间过宽时报告 inconclusive，不改变门槛。
- [ ] 初次正式结果失败需修复时，保留该轮 complete/failed 记录并返回所属阶段；新增十六条独立场景形成下一版正式集，其余八条作为固定公开/历史对照，公开与新增子集分别报告。
- [ ] 提交 `test: freeze candidate quality and blinded comparison evidence`。Git 保存小型汇总/散列/原始评分，不保存全部大模型张量。

**Acceptance:** Q 全部技术与人工子项都通过；“72 个通过”必须写出模式和分母，不能用 24 个成功 prompt 遮住 3 seed 中的失败。

## Task V2: 正式端到端与资源对照

**Files:** Extend `tools/compare_ports.py`, `tools/port_metrics.py`, `tests/test_port_metrics.py`；证据目录与质量运行分开。

- [ ] 核对 V1 之前已冻结的开发集校准结果、双方精度与实际默认模式，直接使用同一候选。不能看过正式 72 例后挑模式；若配置改变，该轮保留为失败/回归资料，重新冻结候选并按独立集规则验收，不混用版本。
- [ ] 执行六个 performance case，双方各一次预热、每 case 五个 AB/BA 新进程配对，共 60 次正式生成 + 12 次预热；所有质量检查已证明的默认路径都包含完整校验、文本、PE/encoder（相应 case）、去噪、VAE和 PNG 写出。
- [ ] 详细 trace/内存插桩关闭，父进程计时覆盖完整执行。原始记录保留温度、时钟、驱动、CPU线程、电源配置、退出码、实际图片 hash，不能只保存平均值。
- [ ] 同六个 case 另跑统一内存插桩，默认三组配对重复，以每次进程峰值的 case 中位数计算比值。双方一律覆盖设备权重/blob/staging/cache 的实际 Vulkan 分配，并记录 CPU peak RSS；已知无法覆盖的部分使 M 为 incomplete。
- [ ] 每项先求 candidate 中位耗时 / reference 中位耗时，再取六项比值几何平均；S 要求总体 ≤.75、每项 ≤1.05。报告各项数据和配对 bootstrap 区间，不能把“耗时降低 25%”误写成“速度提升 25%”（对应 throughput 至少约 1.33×）。
- [ ] M 分别求 RSS 和进程 Vulkan 分配峰值的六项几何平均比；至少一个 ≤.80、另一个 ≤1.05，每个 case 的两项比值也均 ≤1.05。candidate 每次有效运行实际 device peak ≤6 GiB、RSS ≤20 GiB；超限不能藏在中位数里。
- [ ] 预先记录干扰判据：系统休眠、电源/性能档切换、驱动重置、观察到其他 GPU 计算任务或非实验 CPU 连续 5 秒超过一个核心。仅外部干扰可以使整对无效并补测；自身计算造成的热稳态/降频与资源失败是结果，不因较慢删除。每个 case 最多补两对，再不足则 incomplete。
- [ ] 若参考失败或质量不符合共同要求，速度比 unavailable，保留在六个任务分母；若某项未达 S/M，记录瓶颈再返回 P2，不能换成容易取胜的任务。
- [ ] 提交 `bench: record paired runtime and process memory comparison`，保持实际测量和目标分离。

**Acceptance:** 所有六项都有有效配对且质量可比，S/M 硬门槛全过。单机结论限定已记录硬件，不推广到其他 GPU。

## Task V3: 自动检查证据并形成最终报告

**Files:** Create `tools/evaluate_port_results.py`, `tests/test_port_acceptance.py`；Extend `docs/FUTZ12-COMPARISON.md`, `docs/ROADMAP.md`, `README.md`；小型证据写 `artifacts/<执行日期>/reference-acceptance-v1/README.md`, `results.json`, `sources.json`。

**Interfaces:** `evaluate(targets: dict, evidence: dict) -> dict` 返回每项 `pass/fail/incomplete`、证据地址/散列及原因；只有 F/Q/S/M/D/A 全为 pass 才返回 `surpassed=true`。新工具命令为 `tools/evaluate_port_results.py --targets docs/superpowers/plans/surpass-acceptance.json --evidence DIR --output NEWDIR`。

- [ ] 先写参数化行为测试，覆盖：missing baseline、少一个 seed、不同初始 noise、权重等价未证实、阶段精度不匹配、关掉 PE、图像质量失败、内存超限、未真实执行 Windows、没有公开可获取资产、人工票缺失和版本不一致；每种都必须使 `surpassed=false`。测试使用小型完整 evidence builder，不手填一个 `passed=true` 代替原始指标。

```python
def test_missing_reference_cannot_pass(self):
    evidence = valid_evidence_fixture()
    evidence["performance"][0]["reference_seconds"] = None
    result = evaluate(targets_fixture(), evidence)
    self.assertFalse(result["surpassed"])
    self.assertEqual(result["gates"]["S"]["status"], "incomplete")
```

- [ ] 验收器重新核算全部指标，检查 manifest/hash/分母/限值与实际运行日志；手写摘要布尔值不能覆盖原始失败，缺失数据不默认零误差。
- [ ] 以 D3 的真实 Linux/Windows 离线生成、macOS 小模型、授权后的公开下载检查完成 D。尚缺硬件/评审/发布时清楚列出 pending，仍交付本地完整候选，不把外部条件编造成已完成。
- [ ] 检查所有新增模块的依赖方向、公开 stdlib-only API、无 PNG 的外部链接、schema-1/2 compatibility、CLI旧命令，运行最终候选适用的 CTest/Python/安装检查。此前通过且未受修改影响的昂贵大模型运行复用冻结证据，不反复跑满机器。
- [ ] 更新功能对比表为“能力/本项目实证/对方实证/限制/来源”；报告包含六个 case 的速度/RSS/device 内存、质量/盲评、环境和失败清单。无证据的维度写未验证，不用主观评分填空。
- [ ] 完成提交并记录最终 commit/tag/包 hash；只有验收器全绿后才能写：“在固定 RTX 4060 Laptop 8GB / Linux / Turbo 范围内，功能与交付通过、质量不劣，端到端耗时降低至少 25%，至少一项峰值内存降低 20%。”更广声明另有证据。

**Acceptance:** 用户拿到可复核的功能、质量、性能、内存、平台与结构结论。任一门槛未满足，最终结论必须是对应范围尚未超过。

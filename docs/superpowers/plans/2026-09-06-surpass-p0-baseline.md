# P0 基线与测量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 得到真正运行过、输入与权重身份明确的对方基线，冻结正式样本和公平比较协议。

**Architecture:** 将不可变实验清单、两个程序的适配器、运行驱动分开。每个程序独立进程、独立目录和自己的 ncnn 版本；适配器只改参数/布局，不改模型计算。

**Tech Stack:** Python 标准库/NumPy、现有 `.venv`、固定 Git/Hugging Face 资产、CMake。

## Global Constraints

继承 [总计划](2026-09-06-surpass-reference.md) 的全部约束及 [验收目标](surpass-acceptance.json)。以下新增命令是实施后的接口约定，当前还没有对应程序；不得把命令列在计划里视为已经运行。

## Task B1: 冻结输入和比较契约

**Files:** Create `tools/acceptance_manifest.py`, `tests/test_acceptance_manifest.py`, `tests/fixtures/port-corpus.json`。读取 `sources.lock.json`、现有 artifact、对方固定版本的五个 prompt 文件。

**Interfaces:** `freeze_inputs(spec: dict, output: Path) -> dict` 写入原始 prompt/latent 和 SHA256/大小；`verify_inputs(manifest: dict, root: Path) -> None` 拒绝缺失、改变或重复 ID。输出 schema 包含 `id, split, prompt_source, prompt_sha256, noise_sha256, shape, steps, pe, dtype_by_stage, model_identity`；图生图另包含 `input_image_sha256, decoded_rgb_sha256, strength, resize_policy`。性能 PE 明确为相同 greedy/长度/模板参数。缺失字段不默认为兼容。

- [ ] 写清单行为测试：冻结后修改 prompt 的空白或 latent 最后一字节，核对必须失败；正式 ID 重复也失败。例：

```python
def test_modified_prompt_is_not_same_case(self):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        case = {"id": "changed", "split": "development", "prompt": "a red apple",
                "prompt_source": "unit-fixture", "shape": [64, 64], "seed": 42,
                "steps": 8, "pe": {"enabled": False}, "model_identity": "unit-fixture",
                "dtype_by_stage": {"text": "fp32", "dit": "fp32", "vae": "fp32", "scheduler": "fp32"}}
        manifest = freeze_inputs({"cases": [case]}, root)
        (root / "changed/prompt.txt").write_bytes(b"a red apple ")
        with self.assertRaises(ValueError):
            verify_inputs(manifest, root)
```

- [ ] 执行 `.venv/bin/python -m unittest discover -s tests -p 'test_acceptance_manifest.py' -v`，先确认行为测试失败，再实现不可变输入保存和验证。
- [ ] 将三条历史原文固定为 apples、40-token 英文、32-token 中文；五条对方 prompt 记录固定 raw URL、原始字节 SHA256 和 token IDs，正文留本地实验目录。
- [ ] 将十六条独立正式场景写成完整提示词，主题依次为：中文书店招牌、英文咖啡菜单、日文车站指示牌、三杯不同颜色饮料、左右物体关系、前后遮挡、四格同角色故事、俯视房间布局、夜间低对比街景、金属玻璃静物、自然光人像、运动人物姿态、线描建筑、水彩花卉、双语信息海报、超过 1024 tokens 的多约束室内场景。每条写明确对象、文字及数量，不使用生成时随机扩写。
- [ ] 先用官方与 native tokenizer 计算长度；边界用实际 token 数验证，不能按字符数估计。对 2049 超长输入保存完整原文和双方截断/拒绝结果，容量策略不隐藏。
- [ ] 按总计划生成 72 个正式 case；另外固定 8 个开发 case、6 个性能 case，和三个输入图像 × 五个强度的 15 个图生图契约 case（照片、文字、几何色块，来源清晰）。图像解码 RGB bytes 也冻结，不能等 P3 看到结果后再选图。建立小清单摘要与大文件清单；所有大文件留 `outputs/port-corpus-v1/`，Git 只保存来源/散列。
- [ ] 运行行为测试通过，冻结 `outputs/port-corpus-v1/manifest.json` 与 `protocol.json`，本地提交 `test: freeze cross-port input and acceptance contracts`。

**Acceptance:** 72 个正式 case、6 个性能 case、8 个开发 case；输入身份全部可验证；正式集结果尚未用于调参。正式的长提示词场景具有真实 1025..2048 token 长度。

## Task B2: 运行对方与权重身份审计

**Files:** Create `tools/port_adapters.py`, `tools/compare_ports.py`, `tools/audit_port_weights.py`, `tests/test_port_adapters.py`。复用 `tools/package_model.py` 的校验与 `tools/source_inventory.py`。对方独立源码放 `outputs/reference-port-v1/source/`。

**Interfaces:** `PortAdapter.command(case: dict, output: Path, trace: bool) -> list[str]`；`canonical_latent(raw: np.ndarray, layout: str, shape: tuple[int,int,int]) -> np.ndarray` 返回 FP32 CHW；`verify_pair(left: dict, right: dict) -> None` 校验 prompt、噪声、尺寸、steps、PE、权重与每阶段模式；`audit_port_weights(source: Path, official: Path, output: Path) -> dict` 记录逻辑权重对应关系和规范化流散列。`compare_ports.py --manifest FILE --suite calibration|quality|performance --output NEWDIR` 只生成新目录；`--development` 允许功能尚未完成的 case 保持 incomplete 并报告调优观察，但从不产生最终胜出。

- [ ] 用唯一编号张量验证两种 dump 布局的可逆映射，再验证一次真实输入 dump；测试不以形状相等替代元素位置相等。

```python
def test_hwc_to_chw_preserves_channel_identity(self):
    chw = np.arange(128 * 3 * 2, dtype=np.float32).reshape(128, 3, 2)
    raw = chw.transpose(1, 2, 0).copy().ravel()
    actual = canonical_latent(raw, "HWC", (128, 3, 2))
    np.testing.assert_array_equal(actual, chw)
```

- [ ] 运行 `test_port_adapters.py` 的失败测试；实现显式参数映射、不同日志的解析及格式转换。两侧 initial dump 的 canonical SHA256 必须相同。
- [ ] 拉取固定对方 commit 与其原 submodule，从 `src/CMakeLists.txt` 构建，保存编译器、依赖和二进制散列。平台兼容补丁单独保存，禁止把我们的算子补丁加进对方基准。
- [ ] 下载对方固定模型 revision，校验完整资产；逐层建立 GEMM 转置/打包、norm、bias、embedding 与官方逻辑张量的对应，比较全部规范化 FP32 流 SHA256。明确哪些路径已证明同权重。无法证明的结果标为产品对照，不冒称同权重数学对照，且不能关闭 S。
- [ ] 先跑 64×64 和 512×512 的 PE 关闭开发输入，再跑 1024、长提示词和 PE。保存实际 command、完整 stderr、退出码、图像、初始/最终 latent 和资源；对方失败保持在分母内。
- [ ] 预声明对方/候选校准网格：CPU 线程 `{4,8}`、各自确实支持的 FP32/FP16/BF16、对方 low-vram on/off；off 在资源预检明显超出目标时记录不可用而非触发 OOM。将不可支持组合记 unavailable。PE 参数两侧明确设定，不用不同默认值。
- [ ] 以开发集共同质量标准选出对方最快有效配置并冻结；候选对应的每阶段公开精度配置要一致，否则只能作为不计入 S 的补充产品对照。共同数值要求按双方匹配的精度使用固定 gates；本项目额外接受 FP32 严格验收。当前候选失败与尚缺的原生动态尺寸/图生图记 incomplete，不能阻止先建立对方基线，也不能伪造候选已通过。最终候选校准在 P5 正式质量运行前完成。
- [ ] 完成权重等价的小型报告，执行适配器测试通过，本地提交 `feat: add reproducible reference-port adapter and baseline`。

**Acceptance:** 对方完整图像真正生成；双方输入布局/bytes 已验证；模型身份已说明。对方不能运行的原因已被隔离到可复现错误，不能作速度收益。

## Task B3: 冻结测量范围和运行预算

**Files:** Modify `tools/benchmark_pipeline.py`；Create `tools/port_metrics.py`, `tests/test_port_metrics.py`。现有 benchmark 强制 trace、仅接收 FP16/FP32，不能直接作为新正式基准。

**Interfaces:** `summarize_pairs(pairs: list[dict]) -> dict` 返回每项中位数、geomean ratio、无效配对清单；`validate_measurement(record: dict) -> None` 要求完整启动时钟、设备/精度、trace=false、双方质量状态和输入 ID。

- [ ] 写有意义的拒绝测试：一侧少了 PE、开启 trace、只记 DiT 时间、失败或缺记录，都必须无法生成速度胜出判定。

```python
def test_failed_reference_is_not_infinite_speedup(self):
    pair = {"case_id": "oom", "candidate_seconds": 100.0,
            "reference_seconds": None, "reference_status": "oom"}
    result = summarize_pairs([pair])
    self.assertIsNone(result["geomean_ratio"])
    self.assertEqual(result["unavailable_case_ids"], ["oom"])
```

- [ ] 将 BF16、prompt 文件、PE 和图生图字段显式传递；增加 trace 开关，正式计时固定关闭。用同机外部单调时钟覆盖进程启动到文件关闭完成，内部阶段时间仅作解释。
- [ ] 写固定 AB/BA 调度，每个 case 预热各一次，再 5 组新进程配对；保存每次原始记录。缺失 case 时总体门槛为 incomplete，不能对幸存子集计算通过。
- [ ] 用 512/1024/长文本/PE 的校准时间与 trace 文件大小生成预算：正式运行数量、小时数、可用磁盘、单 case timeout。全轨迹逐 case 校验后打包保存并记录 SHA256，磁盘不足先缩小开发诊断集，不删正式失败或减少正式分母。
- [ ] 写出 `outputs/reference-port-v1/baseline.json`、`outputs/port-corpus-v1/protocol.json`，运行对应 unittest，通过后本地提交 `feat: define paired end-to-end benchmark measurements`。

**Acceptance:** P0 交付的是实测基线、可复现输入和协议。正式 S/M 胜出尚未开始；同主机 OS 页缓存冷/热状态没有证明时，只使用准确的新进程/已预热标签。

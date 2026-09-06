# Q2 native Vector CLI/API 集成独立审查

结论：固定新diff内没有发现 Critical/Important 缺陷；两项 Minor 需收窄身份/证据措辞。保存候选条件的图像诊断真实成立，但不能当作新CLI原生整链验收或默认模式升级证据。

## 固定审查范围

开始时冻结 `task-Q2-vector-review.diff`，SHA256 `ef6492e9246686a0a47d7d9420713bc4a1576580690305f8e27b75fe377a4205`。只审该差异内七个文件：include/ernie/pipeline.h、cli/options.cpp、src/pipeline.cpp、tools/validate_pipeline.py、tests/test_cli.py、tests/test_request_validation.cpp、tests/test_trajectory_report.py。差异中的pipeline blob为653b274→6876a32，validator为421988e→a4f162e，避免将之后schema3和img2img工作混入结论。组件边界只溯源d9f3b4b，不重审其完整数学实现。

## Minor M1 — 把归档脚本数量写成实际导入数量

`artifacts/2026-09-06/text-vector-trajectory/README.md:5`称“实际导入的62个Python脚本”。冻结worker的validator只是`Path(__file__).resolve().parent.glob('*.py')`复制整个脚本目录，不记录sys.modules或导入事件，因此证明的是62份脚本已归档并逐份验证SHA，不是62份全部执行/导入。bootstrap确实把冻结snapshots目录置于sys.path首位，实际相关本地依赖能从该目录导入；该事实不能推出整个目录全部被导入。

建议改为“worker使用的冻结脚本目录共62份文件，已逐份核验SHA”；若要声称实际导入集合，另记录模块解析路径。此项不推翻实际图像边界结果。

## Minor M2 — 文本被绕过时仍记录Gemm reduction

固定diff中`tools/validate_pipeline.py:277`无条件写`text_down_reduction = vector if args.text_down_vector else gemm`。因此合法的`--diagnostic-embeddings`/`--reference-embeddings`流程会记录gemm，尽管本次没有执行native text encoder。conditioning_source与eligible字段正确阻止原生验收误用，但新增reduction字段不应含糊地表示“执行过Gemm”。

建议在两种embedding bypass情况下写`bypassed`或null；或者明确命名为configured_text_down_reduction，并另记录executed=false。已有512归档生成早于该新增字段，因此其历史result没有这个错误字段。

## 模式/公共API审查通过项

- request尾部追加bool，默认false，旧字段顺序与旧源代码聚合初始化兼容；不把此结论扩大为跨版本二进制ABI兼容。
- CLI只在显式`--text-down-vector`设置true，默认Gemm；CLI及C++ validate_request均拒绝与外部embeddings组合。不能通过API绕过CLI约束。
- generate在模型配置后明确拒绝32或其他未审查bucket，只允许64/2048；run_text_blocks Vector组件另检查CPU FP32布局/完整图允许列表。25层调用没有被新flag缩减为部分层。
- CPU文本FP32和DiT所选precision是不同阶段；Vector flag与FP16/BF16 DiT组合不等于Vector文本采用低精度。现有text_device仍只允许CPU。
- validate_pipeline将flag传入实际native command；reference oracle保持官方路径。与reference embeddings、diagnostic embeddings、reference-only的组合都被拒绝；PE可与Vector共存并仍走原PE reference对照。
- 独立AST最小测试枚举16种Vector/reference/diagnostic/reference-only/PE组合，全部符合预期；没有导入torch或启动模型。

## 实际证据独立核验

- artifact/candidate-result.json与outputs/pipeline512x384-pe-vectordown-fp32-v1/result.json逐字SHA相同。
- artifact/trajectory.json与outputs/trajectory-text-vector-v1.json逐字SHA相同。
- 512 runner实际SHA匹配result与worker-snapshot：7d9ea0d6dd6633781f17671bc5c4fa2f8fbf01d5b88d6e2c114e21175168b06c。
- 62份run/scripts逐份SHA均匹配result.source_snapshot，且该字典与worker-snapshot.sources完全一致。validator身份为90ca14d681b1aa336ecbb837ef4734b62e0a222d72d0c6287591ce6896dfcaca。
- 315-token候选实际output SHA为433ae46eaa1c8b831fa273f68b8345646d6ddfdb56695b4d3286b4964bca294a；q2-native-text-vector-2048-v1/output.f32也为同一SHA，说明新原生文本组件单独输出与保存候选逐位一致。这不等于已执行新CLI的整链模式。
- 用当前diagnose_trajectory.summarize_run对两份真实保存轨迹重新核验门槛、输入/输出、runner/script身份并复算，CPU2、没有模型/GPU运行：

| 保存轨迹 | 张量通过 | 失败边界 | PNG MAE/max | overall | native_acceptance_eligible |
|---|---:|---|---|---|---|
| 512×384 saved-vector | 25/25 | 无 | 0.001559787326388889 / 1 | pass | false |
| 中文1024 saved-vector | 22/25 | prediction-6、prediction-7、decoded | 0.022527376810709637 / 13 | fail | false |

中文decoded最大误差0.10383453965187073，参考max1.300978660583496，按固定FP32阈值0.0002+0.01*reference_max约0.0132097866判为失败；不能因其NRMSE约0.00102033通过而忽略max失败。512 decoded max0.0057846009731292725低于固定0.01099487681388855。没有观察到后验放宽门槛。

原生历史512 PE→text→image 24/25且decoded失败仍在轨迹中保留。正式资格是`native_acceptance_eligible=false`，不是语义相反的“ineligible=false”。本次两个保存输入实验均不能代替新flag下的完整PE/text/DiT/VAE验收；中文失败进一步阻止将Vector提升为默认或宣称广泛质量闭合。

## 验证边界

本审查没有build、没有完整模型运行、没有GPU推理、没有改实现或历史证据。两次saved trajectory复算只读取已有张量。没有把原有工具之外的潜在泛化问题列为当前新增diff缺陷。后续schema3/ComponentFiles加载器胶合需独立审查，本报告不覆盖它。

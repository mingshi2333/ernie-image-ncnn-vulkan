# F1 1376×768 官方 VAE reference-only 实际执行

状态：`actual_official_reference_completed_pending_independent_review`。只完成固定 latent [1,32,96,172] 的官方CPU decoder参考；native/head/DiT及生产registry没有执行或修改。实际输出 `outputs/f1-shape1376-s64-plan-v5/official-vae`，控制器状态 `completed_reference_not_native_validated`。

## 授权入口与失败保留

执行前外部完整SHA核对plan `bae415829ab5a6a8bf717e0f112085b9389e4f81a701ecaccaa4d2b060aaa2b5`、冻结controller `360e7f7c6a76c41d56c1319c956862953e2ab3e02bd35e9d16e101e3ffebba21`；只执行plan.launcher_argv，未裸跑worker或替换guard。

首次外层systemd-run因工具环境缺少XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS，0.002518s退出1。`external-launch` 完整保留；控制器与模型均未启动。root明确确认补当前用户已有bus环境属于授权范围，新增 `external-launch-v2` 只向外层启动补 `/run/user/1000` 和 `unix:path=/run/user/1000/bus`，原argv/源码/数值/资源条件不变。

实际unified session98494已exit0，外层parent2213874/launcher2213910，外部wall119.451876776s，日志为空。结束立即通知root释放唯一大型模型槽。后续只做小型CPU证据核对，没有native或第二个模型作业。

## 真实worker与资源

Popen实际worker PID2213979、parent2213910、start_ticks11418437，与四个runtime边界及顶层报告完全一致。entry为冻结export_vae.py，entrySHA `8c13d207d5787c847c56df026617eed63adc95fc743cddb8dafa2e9044d7f368`；venv调用路径和prefix保持工作树`.venv`。actual runtime报告SHA `2761496cd39e715eef40c0173215429aa1ace9d89efd8c4cc98ccc88c3bf71fb`。

before_model / after_model / after_first_forward / after_second_forward 均2822文件、137映射，完整集合与已认证allowlist相同、process身份一致，unknown=[]、error=null、authenticated=true。控制器在worker退出后重新认证实际报告和依赖；独立后处理再次核307个冻结源码、actual/allowlist全字段集合、source/entry/collector身份和报告散列。边界范围不含瞬时load/unload安全证明。

实际scope为 `ernie-vae1376-bdc8f632fe34.scope`，memory.max17,179,869,184B（16GiB）、swap.max0、cpu.max200000/100000、affinity12,14。官方Torch threads2、CUDA隐藏且offline环境。worker过程记录wall113.033615829s，sampled cgroup memory.current峰6,316,261,376B，host minimum13,309,755,392B，memory.events全部0（含max/oom/oom_kill/oom_group_kill）。过程wall含导入、权重验证、模型构造、两次forward及runtime散列，不是纯decoder耗时；scope采样峰也不是精确RSS。外层wall另含scope启动及控制器认证收尾。

## 完整输出分母与数值

输出目录恰三文件 `fixture.json`、`in0.f32`、`out0.f32`，两tensor均标准little-endian FP32，无pnnx/trace导出文件：

| tensor | shape NCHW | elements | bytes | full SHA256 |
|---|---|---:|---:|---|
| in0 | [1,32,96,172] | 528384 | 2113536 | bdd6e7dba29f6113667c83b838af32854031a06c40510f43b555d65bce557f15 |
| out0 | [1,3,768,1376] | 3170304 | 12681216 | e8c5daefae6b065680769560f7a91f10140cd38a9a10dacee3b9ac6a1634b12d |

逐元素有限值检查通过；in0范围[-4.9401478767,4.7572336197]且与冻结saved input逐字节相同；out0范围[-1.0658640862,0.7228025198]。官方 `_decode` 一次，`export_component(Decode(model))` wrapper一次，共2次forward。被冻结的export_component对全输出计算max_abs_error/NRMSE并要求max严格0；实际fixture记录两者均0、reference_max_abs1.0658640862。第二份wrapper输出未另存，所以不声称独立事后重算两份或逐位验证（本轮确立的是执行时全张量数值相等）。

权重实际load_vae先流SHA认证decoder `197d7e112850b5525f70b46af3e899b70199a59d0afdf815fbcfe9e2a0dee6ea` / postquant `9d8a703c17acf65f7f12fdc7cde6e52e73be83fd8cadf51e42737f50afe9f8a9`，按固定prefix严格load_state_dict；官方revision `bc68c81e2a1730a394d5fc9fae70713dee940140`、实际Autoencoder源码SHA `7d9a976c1e4f42615e8c422f1643d86b49c4339221bd04b67f518b718ebd6c2d`、configSHA `4d5ba5e01de06d589dd46e2955ab97e2b0968703dce31b9ebe1d2a38c141836d`保留。只加载decoder/postquant，不是全包权重审计。

`actual-evidence.json` SHA `7591bcbb8113a7a516a01d2ea9d8eac1a9dcf0d3d931c705a916d4a3c469667b` 绑定19文件（含原bus失败/成功外部记录、plan/source/runtime/probe、过程和输出）、两tensor完整统计与分母。所有列项大小/SHA已本地检查。

该reference输入是synthetic unpacked latent，不含上游BN/unpack，也没有RGB→encoder路径。native固定2reshape候选还需full-template/bin身份、实际CPU direct验证及独审；heads/36DiT/完整图像和production registry均未闭合。现将本次成功执行交frozen/root独立复核，之后再准备native候选。

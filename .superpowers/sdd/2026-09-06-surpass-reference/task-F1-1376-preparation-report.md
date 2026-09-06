# F1 1376×768/s64 固定组件 CPU 准备

状态：工具与候选准备完成，尚未执行官方/原生模型。生产 schema3 registry、C++ reviewed whitelist、s2048/6144上限均未修改；没有读取正式72/15样本。后续数值通过前不得注册生产实例。

## 实现

- `tools/vae_shape_contract.py` 提供明确 latent H96/W172 opt-in；普通1..128边界不变，固定入口只允许reference-only，>4096 latent面积始终禁止whole-VAE pnnx。双reshape函数拒绝未知/重复/缺失节点、错误参数和arity；非shape行保留。
- `tools/export_vae.py` 增加 `--fixed-1376x768`、`--threads 1..4`、`--official-root` 和 `--input-f32`。固定实验采用threads2及已经冻结的little-endian FP32有限输入，shape/size/finite验证在权重加载前完成；实际输入SHA从消费的原始bytes计算。默认旧路径保持原行为。reference仍比较真实官方_decode与既有Decode wrapper，reference-only不会导入/调用pnnx转换。metadata明确candidate/ineligible。installed runtime完整冻结仍是执行前条件，不能只以Python可执行文件SHA代替环境身份。
- `tools/specialize_vae.py` 仍要求完整8×8原图固定SHA和同官方weights/revision/config/source的独立目标fixture。新固定入口改为flatten16512与W172/H96；template model必须显式绑定完整bin SHA，specialization前后实际流hash一致，不改权重；manifest重复核验产物bin。它是带symlink的组件验证候选，不是可交付模型包。
- `tools/prepare_shape_1376.py` 只接受72bb…的既有1024/s64完整源manifest。先审核64个源图，再按现有枚举字段生成64份候选图并检查target规范SHA相同，25层text仍s64。输出只有图和计划，没有bin修改或schema3 manifest。所有source bins预期hash来自认证manifest，`weights_content_verified=false`诚实保留，后续每个实际组件必须全流校验。记录fixed4192token，无s2048扩展。

## 已完成的小验证

两个新测试模块共6/6通过；四个工具py_compile通过。另对真实 `models/vae-8x8-v1` 完整graph固定SHA及198,481,196-byte bin实际stream SHA核验通过，纯图变换恰2处，无模型加载。该单bin核验不冒充36DiT等全部weights复扫。

实际准备目录：`outputs/f1-shape1376-s64-plan-v2`。296源码snapshot、64候选图、runner和FP32input摘要已在准备后逐项复核。input shape[1,32,96,172]、2,113,536bytes，SHA `bdd6e7dba29f6113667c83b838af32854031a06c40510f43b555d65bce557f15`。head runner SHA `dbd6d9d057737dc1d7a55f4c4062cc1b85f9653d4de00d2551cc2b212d62caba`，来自已认证D3 Vulkan构建，实际head probe独立decoder允许每轴<=256，故可用H96/W172。旧v1准备保留，后续只选择v2；v1未执行。

## 下一次单组件步骤（全部待root资源放行）

1. 审核v2 `plan.json/source-identity.json`，冻结实际installed Torch/Diffusers/Safetensors运行来源及官方VAE manifests/config，再外部guard设置CPU2、memory.max16GiB、swap0、hostAvailable>=3GiB、timeout1800，串行执行。计划记录guard要求而未声称已经启动guard。不要直接裸跑计划argv。
2. 官方VAE：计划第一步给出snapshot `tools/export_vae.py --height 96 --width 172 --reference-only --fixed-1376x768 --threads 2 --official-root … --input-f32 …`。保留实际source/runtime/input/weight身份，取得完整[1,3,768,1376] FP32参考，旧固定gates不变。
3. 使用同snapshot `tools/specialize_vae.py --template models/vae-8x8-v1 --reference <v2>/official-vae --output <v2>/vae-candidate --fixed-1376x768`；完整模板/bin流hash与参考文件绑定后，`validate_dit_heads.py` 配冻结runner、backend cpu、precision fp32、vae-convolution direct验证全分母。这里只验证unpacked latent→decoder，不包含BN/unpack，仍需后续packed128×48×86→32×96×172及inverseBN eps1e-5边界。
4. Heads分别以packed H48/W86/text64制作官方fixture和native完整输出对照，input head8outputs、output head1output；保持精确权重和全部head图规范SHA。随后一个4192token DiT block同输入官方对照，再36块轨迹。未生成假的官方head/block输出或给未执行任务预填pass。
5. 组件真实正确性+发展整链闭环后，另行审核新schema2完整来源清单及生产registry/shape whitelist；没有在本切片自动批准。资源容量、正式F1、S/M均未评估。

## 准备修正与实际 import-only runtime 冻结

Root发现并复现v2错误：`Path(python).resolve()` 使计划从项目 `.venv/bin/python` 变成 `/usr/bin/python3.14`，两者虽指向相同可执行文件bytes，但prefix/site-packages不同。这是实际依赖来源改变，v2作为未执行的错误准备保留，不再选择。工具现在保留绝对venv invocation，单独记录resolved executable/hash，并以同入口读取prefix、Torch/Diffusers/Safetensors实际origin/version/direct_url；不得以binary SHA替代Python环境身份。

最新计划 `outputs/f1-shape1376-s64-plan-v3`。已核对其invocation仍为worktree `.venv/bin/python`，runtime prefix为该`.venv`、Torch origin在`.venv/lib64/python3.14/site-packages`，resolved仅作记录为`/usr/bin/python3.14`。native验证说明修为实际支持的 `validate_dit_heads.py --cpu-only --vae-convolution direct`。冻结head-runner普通decoder内部num_threads=4保持不变；计划明确official_torch_threads=2、native_ncnn_threads=4、scope_cpu_budget=2/affinity12,14，不能把CPU预算误说成原生内部线程数。

新增 `tools/vae_reference_scope.py`：本次只执行capture-runtime，导入未来官方worker依赖但不加载模型。冻结2822个实际已导入文件/当前映射文件的大小与流SHA（含137个mapped files），复制Python源码到runtime/sources；记录真实package元数据、prefix。此scope明确仅实际imported/mapped集合，后续lazy-load新增依赖需要另外记录，不能声称整个环境完全封闭。源码identity另绑定官方VAE config和decoder/postquant manifest SHA；未来load_vae仍全流核验实际官方权重。

v3 plan给出具体systemd-run scope launcher：16GiB/swap0/CPU200%、affinity12,14，保留venv invocation、CUDA_VISIBLE_DEVICES为空、禁HF联网并固定线程环境。guard在模型运行前认证源/runtime/input/官方metadata和实际cgroup配置、venv prefix、affinity；只接受固定96×172/reference-only/threads2 argv；借用已审check_release.execute的50ms hostfloor/timeout监控，完成后核runtime/input/metadata及OOM，保留独立process/result。guard/launcher未执行，模型输出仍不存在。

新增runtime文件同size篡改及venv路径语义小测试2/2，其余6/6重跑通过、工具py_compile通过。v3源/input/runner/runtime identity绑定已复核；只做了轻量runtime导入与流哈希，未进行官方forward/native推理/导出/GPU。

## 6167d50 Important I1 修复：实际 worker runtime 认证（新 v4，未执行模型）

旧 `outputs/f1-shape1376-s64-plan-v3` 保留。新增 `outputs/f1-shape1376-s64-plan-v4`，plan SHA `5d38b7e0847ba1859b0127ca809c59389540bbf35e7e1552fa13beed4f87a4b0`；305 个 source snapshot 完整 SHA 已验证，导入准备仍为 2822 文件/137 mapped 文件。session 81433 已 exit0，只执行导入准备，无模型构造、forward、pnnx 或 GPU。

实际 `export_vae.py` 的 guarded reference-only 路径新增 `--runtime-identity` / `--runtime-report` 成对参数。WorkerRuntime 在 before_model、after_model、after_first_forward、after_second_forward 四个真实模型进程边界采集 sys.modules 文件及 `/proc/self/maps` 中实际文件映射，记录真实路径、完整流 SHA、大小和每边界全量清单。固定 allowlist 来自独立导入准备；已知文件必须与其一致，同文件跨边界变化/消失也失败。新增文件归档到实际报告目录的 SHA objects 后显式列 unknown，不能自动扩充授权。即使两次 forward 已完成，只要 unknown/changed/缺边界，身份仍为 unaccepted，控制器不能报告 completed_reference。

监督器要求新的 actual report 存在、四边界完整、所有实际文件属于原 allowlist、report union 一致、collector source SHA 与冻结源码一致、venv invocation/prefix 相同，并重新检查实际依赖文件完整 SHA。实际报告 SHA 写入 supervisor result；模型本身发生异常时保留原异常与 partial runtime 报告。新 runtime 未认证时可能已留下 fixture，但这些输出不能用作已认证参考，也不能晋升生产 shape。

这解决的是实际模型进程的**边界依赖身份**；不是声称能观测两个边界之间瞬时加载后卸载的所有代码。准备清单仍诚实标 import-only，不冒称真实两次 forward 的全集。未知依赖必须经独立审查和新准备授权，不能从运行结果自动认可。

保留原 venv 入口、两次 decoder forward、no pnnx、latent [1,32,96,172]、official threads2/native ncnn threads4/scope两核，16GiB/swap0/CPUQuota200%/affinity12,14/hostfloor3GiB/timeout1800。没有实际模型授权或执行。

验证：12 个小型 CPU tests 通过（既有 shape/plan6 + runtime6）。新增测试验证实际新 sys.modules 文件能被枚举；完整四阶段允许项通过、删阶段拒绝；新增模拟 mapped binary 留存原字节但拒绝认证；同尺寸源码变化拒绝且保留失败报告。Python compile 检查通过。新 v4 所有 source/input/runtime identity hash 与计划一致，worker-runtime 目录尚不存在。等待独立复核后再由 root 排队模型。

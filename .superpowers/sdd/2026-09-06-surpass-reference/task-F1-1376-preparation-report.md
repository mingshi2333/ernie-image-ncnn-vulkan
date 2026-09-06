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

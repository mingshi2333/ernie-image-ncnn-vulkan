# ncnn 最新 Git 数值回归（2026-09-08）

## 版本与范围

测试项目源码为 `2490136ee8e11a2cf4702180d3938d3bb1f6086d`，Linux / RTX 4060 Laptop 8GB。正式依赖仍锁定 `6a1bf000f363714839a36793addc8c879d3d899e`；候选是测试开始时的上游 master `3b7bdba7fc8aea8fd46779533eee027df77c639d`，领先 8 个提交。候选通过既有 `ERNIE_NCNN_SOURCE_DIR` 与 `ERNIE_ALLOW_UNPINNED_NCNN=ON` 单独构建，没有修改原 ncnn 子模块、模型包、运行默认值或数值门槛。

两端使用 Clang Release、相同 ncnn 编译选项与同一 glslang 提交 `780b6a209132dc74e99458f62ebe5fa7e57c41cf`，compact reader 和分配统计均关闭。两份 CMake 配置、冻结二进制与项目源码归档的 SHA256 分别在 [build-configurations.json](build-configurations.json) 和 [frozen-identity.json](frozen-identity.json)。本次没有 Windows/macOS 候选构建、性能基准或广泛画质验收。

## 已有舍入修复与此次新增修复

- [BF16 舍入修复 #6925](https://github.com/Tencent/ncnn/pull/6925) 已在当前锁定版中；锁定版领先其合并提交 9 个提交。
- [FP8/FP16 转换舍入修复 #6958](https://github.com/Tencent/ncnn/pull/6958) 也已在锁定版中；锁定版位于其后 1 个提交。
- 候选额外包含 [Vulkan BF16 packed 兼容修复 #6962](https://github.com/Tencent/ncnn/pull/6962) 和 [Reduction 共享累加精度修复 #6964](https://github.com/Tencent/ncnn/pull/6964)。所有新增提交及变更文件见 [upstream.json](upstream.json)，前述祖先关系见 [rounding-ancestry.json](rounding-ancestry.json)。

## 构建与项目测试

候选完整构建成功，原有着色器身份检查通过；两份注意力着色器及 modelbin 源码保持相同，[source-compatibility.json](source-compatibility.json) 记录了两版完整散列。串行运行项目 CTest，**57 通过、0 失败、0 跳过**；包括真实 NVIDIA Vulkan 上的 FP16/BF16 残差、归一化、GELU，以及权重读取/放置、缓存和本机 SDK 消费测试。详见 [ctest.xml](ctest.xml) 与 [ctest.log](ctest.log)。这些是无大模型测试，下面另列真实权重与完整生成。

## 转换与 Reduction 最小对照

独立探针 [precision_probe.cpp](precision_probe.cpp) 使用同一源码分别链接两份 ncnn。10 组标量输入分别检查 FP16 和 BF16 的预期位模式，两端均全部通过。测试覆盖正负数、正负零、应当向上舍入的普通值与跨指数进位；不声称覆盖所有 NaN、次正规数或正好处于中点的硬件舍入规则。

直接调用 Vulkan Reduction，显式把上传数据转换到该算子要求的 pack1，以 FP64 计算参考值。每种精度测试 4 组输入，共 12 项；所有输入都可被 FP32、FP16、BF16 精确表示，避免输入量化与累加误差混在一起。旧版 11/12 通过，候选 12/12 通过。

能复现修复的输入为 4096 个数，除 `x[0]=256`、`x[1]=1`、`x[32]=-256` 外全部为零。正确均值为 `(256 + 1 - 256) / 4096 = 0.000244140625`。

| 精度 | 旧版输出 | 候选输出 | FP64 参考 |
|---|---:|---:|---:|
| FP32 | 0.000244140625 | 0.000244140625 | 0.000244140625 |
| FP16 storage / FP32 arithmetic | 0.000244140625 | 0.000244140625 | 0.000244140625 |
| BF16 storage / FP32 arithmetic | **0** | **0.000244140625** | 0.000244140625 |

旧版将共享部分和 257 写成 BF16，舍入后成为 256，再与 -256 抵消。候选用算术精度保存共享部分和，保留这一个单位。这是在本机实测到的算子改进；它不是本次新增的 FP16/BF16 转换舍入修复。原始逐项结果见 [旧版](precision-baseline-v3.jsonl) 与 [候选](precision-candidate-v3.jsonl)。探针最初缺少直接 Vulkan 调用所需的 packing 适配，未进入有效比较；修正及全部尝试边界记录于 [probe-setup-history.json](probe-setup-history.json)，没有将其计为 ncnn 或项目回归。

共享模型中 11 个文本格式图对象未含通用 `Reduction` 层，见 [model-graph-scan.json](model-graph-scan.json)。项目的 FP32 残差、归一化封装、补偿注意力与 CPU VAE 仍有各自职责；不能因为通用 Reduction 修复就删除这些处理。

## 真实 DiT 权重：9 组旧版/候选配对

每组相同模型、同一组输入、Vulkan device I/O、host weights、repeat=1，使用原 fixture 中冻结的门槛。这里是实际权重配合合成隐藏状态/调制向量的单块测试，不能替代自然提示词完整推理。9 组共 18 次执行，所有调用均返回 0；每端合计 **58,195,968** 个输出元素。9/9 组输出 SHA256 完全相同，旧版到候选的最大差均为 0。

| 输入 | 两版 NRMSE | 原 NRMSE 门槛 | 对官方单块参考 | 旧版/候选 |
|---|---:|---:|---|---|
| s288-fp32 | 2.90843087e-06 | 2e-05 | 通过 | 逐位相同 |
| s288-fp16 | 0.00121938754 | 0.01 | 通过 | 逐位相同 |
| s288-bf16 | 0.00975503797 | 0.03 | 通过 | 逐位相同 |
| s4160-fp32 | 2.9570912e-06 | 2e-05 | 通过 | 逐位相同 |
| s4160-fp16 | 0.0012435392 | 0.01 | 通过 | 逐位相同 |
| s4160-bf16 | 0.0099617252 | 0.03 | 通过 | 逐位相同 |
| block31-s288-bf16 | 0.0313349595 | 0.03 | **未通过** | 逐位相同 |
| block33-s288-bf16 | 0.0318976121 | 0.03 | **未通过** | 逐位相同 |
| block35-s288-bf16 | 0.0403360802 | 0.03 | **未通过** | 逐位相同 |

第 31、33、35 块的既有 BF16 未通过项仍存在，候选没有改善这些误差。完整模型/输入/执行文件散列、最大误差、全局最大误差门槛与日志路径见 [block-results.json](block-results.json)。没有放宽容差。

## 完整 512×512 生图

5 次新生成全部完成：候选 FP32 一次，FP16/BF16 各做旧版与候选配对。FP32 控制复用已经提交并重新认证的历史完整基线；因此不是本轮新跑了两次 FP32。提示词为 `A red apple on a wooden table, soft daylight, realistic photo.`，15 个实际 token、32-token 文本来源、64 个 DiT 文本槽、8 步、相同初始 FP32 噪声、无 PE。三种精度均保留 CPU 原生文本、Vulkan DiT、FP32 残差/归一化保护和 CPU direct VAE；2 线程、host weights、stdio、cache0、trace ON。

初始噪声、提示词、模型、官方参考及全部精度门槛在运行前冻结于 [full-plan.json](full-plan.json)。历史 FP32 控制的 25 个散列从已提交的证据取出并重新核验，见 [historical-fp32-anchor.json](historical-fp32-anchor.json)。此轮没有重新运行官方模型，也不是新的离线隔离或跨平台执行验证。

**三种精度的旧版/候选全部 25 个张量和 PNG 文件均逐位相同。** 每次生成比较 4,981,760 个元素；5 次新生成合计 125 个张量、24,908,800 个元素。全部形状和有限值检查通过，PNG 与解码张量的量化结果精确一致。完整逐项结果见 [full-comparison.json](full-comparison.json)，精简汇总见 [summary.json](summary.json)。

下表是两版相同的官方参考误差，PNG 按 0–255 的 RGB 通道值计算；每种精度使用自己原有的冻结门槛。

| 精度 | 官方张量门槛 | PNG MAE | PNG 最大差 | 原数值验收 | 旧版/候选 |
|---|---:|---:|---:|---|---|
| FP32 | 25/25 | 0.000361124674479 | 1 | 通过 | 张量与 PNG 逐位相同 |
| FP16 | 23/25 | 0.23598353068 | 109 | **未通过** | 张量与 PNG 逐位相同 |
| BF16 | 18/25 | 1.16504923503 | 98 | **未通过** | 张量与 PNG 逐位相同 |

FP16 未通过 `prediction-7`、`decoded` 和 PNG 最大通道差门槛。BF16 未通过 `prediction-5/6/7`、`step-7`、`final`、`unpacked`、`decoded` 和 PNG 最大通道差门槛。两版上述误差完全相同；没有放宽容差，也不以视觉相似代替数值验收。其他中文、长提示词、尺寸和图生图旧结果没有因此更新。

5 次进程都正常退出；各自 cgroup 的 max/OOM/OOM-kill 均为 0。整张 GPU 的采样峰值在 2238–2502 MiB 之间，包含其他进程，不是本模型的分配器峰值。这里有逐步 trace 和 CPU 配额，只有一次各配置执行；耗时及资源记录用于说明本次运行，不能用来宣称提速或普遍内存收益。每次的命令、生成报告和资源结果在 `full/`。

执行结束后重新核验全部 124 个模型、噪声、官方参考和二进制绑定，大小与 SHA256 全部一致。原始 ncnn 子模块保持干净且仍在 `6a1bf000`，正式锁定、默认值、数值门槛和所有历史未通过项不变。

## 对当前项目的结论

最新 Git 在本机通过了兼容性回归，并在独立 BF16 Reduction 抵消测试中产生了可验证的精度改进；当前项目这一个完整提示词样例的 FP32、FP16、BF16 图像没有改善或退步。两项转换舍入修复原本已被项目包含，不能再次计算成候选的收益。暂时保留生产锁定版，候选可执行文件用于本地试用；本轮不修改原三平台 CI 结论。

附带枚举到的两套本机 Mesa llvmpipe 仍报告 BF16 packed 可用、BF16 storage 不可用，见 [environment.json](environment.json)。packed 兼容修复不会让驱动凭空增加原生 BF16 storage 能力；本机枚举并不是重新运行托管 CI。

## 复现实验

本机完整实验位于 `/var/tmp/ernie-ncnn-latest-20260908-v1`。该目录保存两份冻结生成器、单块探针、ncnn 候选源码、独立 build、输入绑定、全部输出与进程记录；权重、构建产物和图片未加入 Git。配置命令见 [configure-command.json](configure-command.json)，两个执行入口为 [run_blocks.py](run_blocks.py) 和 [run_full.py](run_full.py)，数值复算见 [compare_full.py](compare_full.py)。脚本保存的是本机路径明确的实验复现入口，不是通用安装工具。

候选生成器、单块程序及三张候选图片另保存在项目 `outputs/ncnn-latest-git-v1/`；身份记录为 `build-identity.json`，图片为 `apple-fp32.png`、`apple-fp16.png`、`apple-bf16.png`。

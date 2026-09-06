# D3 v3 新归档与离线执行独立审查

当前结论：准备与安装/构建身份无阻断；完整生成仍由 root 执行，结果待复核。本次只读 CPU 12/14 哈希/元数据审查，没有模型/GPU执行。

## 执行前身份

- 新 case 与 `/tmp/ernie-offline-img2img-1024-v2/case.json` 全键比较：仅 `archive` 与 `inventory` 变化，model、inputs、expected_png、request、hidden_paths、resources 均保持原样。期望 PNG 仍是通过正式固定门槛的发展样本原文件 d426ac…，不是依据本次输出重新确定。
- 新 archive SHA `d5810af124bac9ebc8b02874e03dfb2a2dd0f5f8b1ede6bb807eb1d2265582d1`，11,193,462 bytes；inventory SHA `9081c867244b45ed8b9422314d486242415e82e4816a5353cf77932d1d8b6610`，30,118 bytes。case SHA `a10712f9a85c098a033c1abcf95a1d7d1f5fe69ad825becfb1cb47e8f0213e28`。
- 逐项独立重算 `/tmp/ernie-offline-img2img-1024-v3/preparation.json` 的全部188项SHA；tar全部181 regular payload的大小/SHA/完整集合，以及候选目录和实际解包目录对应文件SHA，全部一致。不重散列大型模型包；model manifest 与原case不变，完整模型前后验证由冻结实际执行流程承担。
- 实际执行 CLI `d366c835fe8d8a7de06db804cae094a4b81b1c64ccdbb005551d8942aee7c267` 与已审新安装前缀/33binary Vulkan build证据相同；checker仍为原已审 `4f959a945ff7496f03084b59832ae47fda0e7b8235c0a1258667ce47a7de57f3`。release.json内嵌 Vulkan build identity/result 与 `outputs/d3-runtime-build-vulkan-v3` 两原文件内容完全相同。
- 源身份仍 `3a112e7b9df9bb5be02ff21529773966a596d6563dfb694bf50892cec9fa3a27`，288源已在无驱动审查全部认证。新 CPU build日志三份和33binary亦独立重算匹配，实际5/5针对CTest无skip通过（pipeline API、request validation、GPU context CPU分支、CLI、installed consumer）。Vulkan6/6及新安装75文件已在 task-D3-missing-driver-review.md认证。

归档保持 local_review_draft、distributable=false、无公开URL，不含模型，既有许可/平台限制未被小合同或准备成功消除。旧v2实际无驱动失败保持历史，v3实际最终PNG及进程/资源结果未在本节宣称通过。

## 实际完成后独立复核

结论更新：**按原冻结 checker，通过这个固定 Linux 离线发展 case**。`result.json` 为 passed_fixed_development_case，launcher 输出同状态；九个子命令均与各自 sidecar JSON 完全一致。isolation/help/diagnose/verify-model/no-gpu-diagnose/generate/verify-model-after exit0，missing-model/existing-output按原规定exit1；所有 failure=null。没有重跑模型。

本审查重新核全部188准备文件运行后SHA，case与preparation摘要绑定、模型manifest原绑定以及目前288冻结构建源码均一致；实际模型完整内容由同一已认证原生CLI在运行前后两次 `--verify-model` 执行并输出 `Model verified`。审查没有另行复扫多GB模型，也不将源码清单检查误说成执行时读取源码。

从原冻结 checker 导入只读 helpers，复算PNG容器尺寸/CRC，并直接比较完整2,891,207字节：实际输出与原expected文件逐字节相同，SHA `d426ac2a82574ed330d1c0f56061f7a5aa335fc2fb87c5bf31083204a50c9e16`。existing-output拒绝后仍是此SHA。固定request实际为1024²、总schedule8步/strength.5四步后缀、CPU text/vector、Vulkan FP32 DiT、CPU direct VAE、exact LF prompt/saved noise/source PNG，不含PE或trace；生成日志四步完成并保存图像，未换PNG gate。

九条实际argv的sandbox前缀均用原checker `sandbox()` 重建逐项比较一致：new net namespace、整项目隐藏为tmpfs、固定准备树只读、只有results可写、原模型只读重绑中文空格路径、clearenv且固定PATH/OMP/OpenBLAS。isolation shell实际exit0验证所有隐藏根是存在的空tmpfs、无VIRTUAL_ENV/PYTHONPATH；输出net:[4026534707]与宿主namespace不同。no-gpu-diagnose实际输出compiled=true/count0/default=-1/error=Cannot initialize Vulkan；它没有再次阻断流程。不是对恶意本机程序的安全沙箱或任意系统环境兼容承诺。

### 资源与通过口径

原checker SHA `4f959a…` 未改变。其固定资源失败条件为host最低3GiB、每命令timeout、指定exit状态、scope OOM/oom_kill/oom_group_kill；它**未把 memory.events max 非零作为失败门槛**。九个记录逐项核对host floor/timeout/exit/OOM均满足，故保留通过结论不涉及事后放宽gate。

- generate：505.089993 s，50ms采样整个supervisor cgroup `memory.current`峰6,489,657,344 B，host min10,233,769,984 B，events max/oom等为0。
- verify-model-after：23.434450 s，仍exit0，但相同scope采样达到10,737,418,240 B（10GiB限额），累计memory.events max=49634，oom/oom_kill/oom_group_kill=0。
- 此scope覆盖supervisor及各串行命令；记录的是每命令采样窗口的memory.current最高观察值，**不是进程RSS或精确allocator峰**。events为累计值。不得写全流程max0或全流程峰只有6.49GB。没有memory.stat/归因证据，不解释为特定RSS、pagecache或模型张量组成。触限压力是保留观察项，不能从未OOM推导无资源风险。

launcher绑定10GiB MemoryMax、swap0、CPUQuota200%、affinity8/10，执行固定checker；本独立审查仅CPU12/14。整体墙时549.33s由root外部会话记录，本result本身提供九个分命令时长而非独立完整父时钟字段。S/M与full_quality_gate均明确not_evaluated，distributable/published仍false；一次PNG复现不能扩展为正式质量全覆盖、性能优势、任意平台或可公开发行。

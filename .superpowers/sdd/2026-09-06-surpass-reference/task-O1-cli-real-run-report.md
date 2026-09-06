# O1 new-hook actual allocator and 64x64 paired preparation

2026-09-06；根授权独占GPU，严格串行。小probe仍不加载模型，完整fixture只使用历史64×64开发用例，不使用正式语料，trace开启用于逐位对照，时间不用于S。

## 已完成：新 hook 实际小 probe

为现有真实allocator合同增加设备身份输出及每个allocator必须能关联设备身份的断言。记录name_utf8_hex、index/vendor/device/API/driver/pipeline_cache_uuid，明确不是deviceUUID。ON/OFF独立增量构建CPU预检成功，3GiB/swap0、-j2。ON构建峰801.1MiB、OFF563.4MiB。

outputs/allocation-real-o1-v3：两份冻结probe实际GPU串行exit0，1024-byte output逐位一致，SHA256 990fabcba00d265c85b37a1d1a73d18c1c0abb87d944584f70000bf2ab62544c。实际同时峰值2134016 bytes，11分配；最后instance_destroyed、live0、all allocators inactive、valid true。设备RTX4060 Laptop/index0/vendor4318/device10464/driver2497102272，pipeline cache UUID a7bb89c3244a5872a383c94fef250d65。result.json引用冻结identity SHA，保留事件、guard样本、日志。

## 已冻结并正在执行：完整64×64

核实 outputs/pipeline64-structure-fp32-v1/result.json passed=true，模型 models/pipeline64-residual-v1/manifest.json SHA 9c14feef90fe8e3d189466ce2cef3fce78b9fdf772ffdacee5afe02b40d90752 与历史一致。对模型233文件/符号链接目标逐个bounded-stream散列，保存resolved path/size/SHA；未复制巨量权重。冻结精确apple prompt、历史15个token IDs及8192-byte FP32 CHW [128,4,4]初始噪声。32 bucket默认Gemm文本，PE关闭，8步CFG1，text/DiT/VAE/scheduler均FP32、CPU2、DiT Vulkan GPU0、VAE CPU direct。

outputs/allocation-pipeline64-o1-v1 保存两份冻结CLI、完整source identity引用、模型/输入identity、每侧精确command和guard。两个command数值选项相同；ON多metrics-json、路径各自独立、trace均开启。父guard monotonic时钟跨进程启动至退出，覆盖cleanup/写图/报告。guard限制18GiB进程树RSS sum、6144MiB wholeGPU、host available至少1GiB、墙钟2400秒，wholeGPU样本并不替代实际allocation peak。

ON完整运行session10312已开始；OFF和完整逐位结果尚未完成。此段是开始时状态，不能当实际完整成功证据。

## 完整配对最终结果

ON/OFF完整原生64×64均exit0，随后全量比较PASS：25个FP32 tensor、IDs/prompt（合计27trace文件）和PNG逐位一致。PNG SHA256 44ea37d9f2f73dc64a4c510d9fc262c7b02dffb689f67010bb34c676025f81ba。实际15tokens与冻结历史IDs一致，未用saved-embedding绕过文本。两侧完整stdout/stderr以合并日志保存；仅移除ON明确diagnostic提示、已知进度行时间及侧输出路径后，全日志一致，不声称单独stdout原始字节一致。

ON metrics：valid/coverage_complete true，2375实际分配，全部同时峰968724992 bytes，退出live0。1254 allocator generations均inactive；初末无全局instance；RTX4060 Laptop真实身份绑定存在。三类memory峰83918848/4002304/880803840 bytes，但以全局同时峰为准，不能累加类别峰当全局结果。清理后各类live0。

父guard ON368.486051889s/OFF368.159374994s，完整monotonic_ns记录覆盖启动、模型验证、native执行、cleanup、写图、metrics写出与退出。trace-on并非正式速度窗口。RSSsum峰ON1119825920/OFF1220976640 bytes，整卡采样峰2688/2845MiB，hostavailable最小17842704384/14867070976 bytes。两侧均未触发18GiB RSSsum、6144MiB整卡、1GiB hostavailable边界。整卡采样不是实际allocator峰，CPU RSSsum也不是PSS。

所有模型233文件在两次运行后再次bounded-stream全SHA核验，bytes/size/resolved path与预冻结一致。小可提交证据在 artifacts/2026-09-06/allocation-cli-pipeline64，完整原始事件/trace/日志继续保留outputs；结果引用冻结identity、完整metrics和证据文件hash。probe和全模型会话均退出，GPU已明确交还root，随后只做CPU封存。O1阶段记账/GPU时间等剩余空项未关闭；此一例不关闭广泛质量、正式S/M或完整O1。

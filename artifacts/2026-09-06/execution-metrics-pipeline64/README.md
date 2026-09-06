# 固定64×64完整生成：执行指标开关验证

状态：`independently_verified_fixed64_instrumentation_invariance`。实现/结果提交0fbd387，独立复核c11ccb3。本次是trace-on诊断；formal speed/memory均false，没有新增官方质量oracle。

实际ON与OFF均使用同一保存latent、prompt、模型及已冻结源码，Vulkan FP32、8steps、CPU direct VAE、threads2，无PE或embedding bypass。两次串行进程均自然exit0，27份trace文件（其中25份完整有限FP32张量）及最终PNG全部逐字节相同。初始latent、prompt和IDs与冻结输入完全相符。PNG SHA `fdd6e29e33800b7ca025d43f488b71499dac1cbfea099e1d96c09f31257e8fd7`；独立完整解码的12,288RGB字节SHA `a7f93dffb2f376d379a17c2307e124ae1ac7ee3e18cc652ec03241ee1d8eb396`。

## 已观测阶段

| 阶段 | Host秒 | 样本数 |
| --- | ---: | ---: |
| 模型校验 | 18.381652 | 1 |
| 合并读取/准备 | 320.354133 | 322 |
| 已观测计算 | 27.458741 | 321 |
| 顶层初始上传 | 0.052921 | 1 |
| 最终下载 | 0.000051 | 1 |

CLI生成/写图总区间367.954317秒，已知阶段和366.247497秒，另1.706820秒未分类。read_prepare包含load_param、load_model内的读取、解码/展开、pipeline创建、权重上传及内部wait，并包含尚未拆分的heads/VAE复合工作；它不是纯磁盘读取。322条由25个text block load、36×8个DiT block load、8个heads复合调用与1个VAE调用构成。没有将推定内部时间填入read/prepare/wait/GPU时间；这些字段、传输字节和CPU RSS保持null。submissions866仅覆盖明确观测的范围。

## 分配与资源

ON实际观测2375次vk_device_memory分配，同时峰968,724,992字节；1254个带generation的allocator结束时全部inactive/live_handles0，总live0。其范围限于被观测的ncnn allocator生命周期，不等于整卡或全部驱动内存。

ON父进程wall388.594904秒，OFF382.261158秒；各自scope采样memory.current峰5,852,770,304与10,161,700,864字节，均低于10GiB硬限。host available最低分别16,308,576,256与16,485,826,560字节；swap/max/OOM均0。样本字段保存累计峰而非瞬时memory.current，未记录memory.stat，不能把两侧差值归因到RSS、文件缓存或钩子节省。单对、trace-on、两核诊断不产生正式S/M结论。

## 证据保存

73个实际证据记录全部在独立复核和本次归档时完整重验。实际输入/完整trace/PNG/两份约50ms采样流及runner留在outputs，由actual-evidence.json和冻结plan记录身份；本目录保存小型计划、执行器、实际结果、过程日志、历史无效准备与清单。原始actual-evidence状态仍为作者复核前措辞，最终独立通过状态由本manifest与c11ccb3审查记录提供，没有改写原始证据。

诊断收集器为显式private调用，默认未启用；已完成步骤在trace下载/写出/回调前记录，后续失败保留已完成阶段，未完成的当前步骤不冒充完整样本。该失败路径有小型合同检查，当前实际大模型样例是成功路径。

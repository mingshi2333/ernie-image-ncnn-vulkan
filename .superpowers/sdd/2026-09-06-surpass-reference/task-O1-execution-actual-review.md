# O1 fixed64 ON/OFF 实际执行独立复核

范围：`0fbd387` 及 `outputs/execution-metrics-pipeline64-o1-v1`。此前准备审查 `3ec6e4d` 已关闭三个 Important，本次核真实执行结果。仅 CPU12,14 只读文件、小型 FP32/PNG 解码及散列；不重新运行模型/GPU，不重建。

结论：此次固定64、trace-on 的 ON/OFF 数值保全与有限范围 execution/allocation metrics 实验通过；没有剩余 Important。正式 S/M 未评估，不能从单对时长/采样峰值推断性能或内存优势。

## 冻结来源与输出身份

actual-evidence SHA `65a89753024d701a68f1ec272fe19f58252a221304cd6f630332d97d1bdff282`，73项实际证据大小/完整SHA全部独立通过。result SHA `1edfdc2a1834b0943110c958aa60c7b39b1f5c115648fe3cb9b11e0ae6769099`。

计划仍是已审 `d20b6b2e…`；launcher/worker/supervisor/compare分别仍为 `3cdd40d7…` / `aee653c4…` / `b1e660e6…` / `3f9555a2…`，没有在观察结果后调整 gate。305个冻结 source 又逐项重新验证通过；两runner保持原冻结身份（ON5abc7b3f…、OFFb597ee4c…）。原 ncnn base/ON派生完整身份、仅allocator hook差异已在准备审查逐文件通过，不把旧trace或旧runner当新实现。

两实际命令除runner、输出目录和ON的metrics-json外参数相同：64×64、8steps、Vulkan FP32、CPU direct VAE、threads2、同模型/prompt/FP32 saved latent；无PE与embedding bypass。worker 到 native 前完整流校验233模型文件、输入/source/runner/cache。两侧成功返回证明其固定前置检查实际执行；本次不另行重复数十GB模型扫描，也不声称存在未记录的全权重事后校验。

独立重新枚举两trace集合，完整分母恰27文件，25个f32全部规定字节数且逐元素finite，所有27文件两侧逐字节相同、SHA与result逐项相同；initial/prompt/IDs又与冻结输入逐字节相同。两日志均真实记录denoise1..8和最终Saved64x64；日志本身含不同时长及ON instrumentation说明，不宣称stdout全字节一致。

PNG完整字节SHA `fdd6e29e33800b7ca025d43f488b71499dac1cbfea099e1d96c09f31257e8fd7` 两侧相同。独立检查chunk边界/CRC，解压IDAT并实现PNG filter1/2/3/4逆变换：64×64、8-bit truecolor，12,288 RGB字节，SHA `a7f93dffb2f376d379a17c2307e124ae1ac7ee3e18cc652ec03241ee1d8eb396`，像素范围0..253。该验证是ON/OFF保全，不新增官方质量oracle。

## 实际资源与时间边界

- ON：wall388.594903989s，7709采样，sampled scope memory.current峰5,852,770,304B，host最低16,308,576,256B。
- OFF：wall382.261158137s，7583采样，sampled scope memory.current峰10,161,700,864B，host最低16,485,826,560B；低于10GiB硬限10,737,418,240B。
- 两scope均实际观察memory.max10GiB/swap.max0，taskset4,6为真实监督命令；两process complete=true/exit0/failure=null/cgroup_seen=true，最终记录memory.events全0（含max/oom/oom_kill）。所有host采样均高于3GiB，最大相邻间隔ON约51.154ms、OFF约51.101ms，最后采样距结束约50ms，无长失察间隙。

样本中的 memory_current 字段是监督器累计峰而非瞬时序列；重算其max及hostmin与process一致。此scope包含worker完整模型hash与native等工作；未存memory.stat，不能把ON/OFF差值归因到RSS、pagecache或统计钩子节省，也不能称全卡显存峰。wall为父进程覆盖scope启动、前置hash、生成、清理/写图/报告和退出等待；与CLI自身计时范围不同。

## 实际事件与阶段语义

ON metrics 与result内嵌metrics完整相同。available/valid/coverage_complete真，initial/final instance均false，真实设备index0为RTX4060Laptop（vendor4318/device10464/UUID a7bb89c3244a5872a383c94fef250d65）。allocation_domain限定vk_device_memory，coverage仅observed_ncnn_allocator_lifetime。

实际记录1254个带generation的allocator身份（other627/staging313/weight304/blob10），2375次分配；总同时峰968,724,992B。allocator与memory-class分配次数分别独立求和都等于2375；全部allocator inactive、live_handles0、live_bytes0，总live0。峰值取事件计数器的整体同时峰，不能把跨allocator峰值相加当整体峰；也不是整卡或完整驱动内存。

阶段统计保留partial_known_intervals：verify1/read_prepare322/upload1/compute321/download1个样本；read/prepare/wait为null，所有GPU时间、upload/download bytes、cpu_rss均null。submissions866仅已观测范围，非全内部提交。五阶段host ns和366,247,497,326，小于CLI host367,954,317,371，差额1,706,820,045未分类；没有因阶段相加超出总时长。计时插入点与不重复合同已由420f6af审查，本次实际计数与其partial设计一致，read_prepare仍含不可拆的加载/准备/上传及heads/VAE复合工作，不能称纯磁盘读取。trace-on、formal_speed_eligible=false、formal_memory_eligible=false保留。

# 默认 FP16 完整回归

本轮补上内存执行公共路径对默认精度的完整回归。命令没有传入 `--precision`，生成报告确认使用 `fp16`；它不是所有参数都采用默认值的运行。模型、保存的初始噪声、15-token apple、512×512、8 步、原生 Vector 文本编码、Vulkan DiT、CPU direct VAE、2 线程与 trace 开启均沿用前例。显式 host 权重，cache/prefetch 0，GPU reserve 512 MiB，buffer auto、spill 2048 MiB、最多 3 次重试。

- 源码：`7296bfeba2c809d54a7d6214b0f08dee861ad1f8`；ncnn `3b7bdba7`。358 个源码文件、159 项绑定、5 个实际派生编译单元。
- 二进制 SHA256：`3202547804114b542a2785a1466bfc117e5d75830bf0633daf7b58cb741910da`，与 v3 相同。
- 计划 SHA256：`80ba498048f0c6a9e29a4fe13a3a908d66f70f64f860c3192fc13c51b519b1cd`。
- 认证旧 FP16 基线：`/var/tmp/ernie-ncnn-latest-20260908-v1/full/candidate-fp16`。
- 完整冻结输入、源码和原始张量/PNG：`/var/tmp/ernie-memory-execution-20260909-v4`；本目录保存可追溯的小型记录，不提交权重、二进制和生成图片。

## 结果与边界

全部 8 步完成，退出 0，零 Vulkan validation/VUID。**25/25 份张量和 PNG 文件与旧 FP16 基线逐位一致**，共 4,981,760 个有限张量值，PNG 与 decoded 的量化精确一致。此次内存改动没有改变该例默认 FP16 的输出。

对同一官方 CUDA FP32 参考仍为 **23/25 张量通过，PNG MAE 0.23598353068033853、最大差 109**。`prediction-7` 与 `decoded` 未通过最大误差门槛，PNG 最大差仍超过原上限 80；平均差未超限不代表整项通过。这些负面结果与旧 FP16 相同，不能把新旧 25/25 逐位一致写成官方 25/25。详见 [完整比较](full-comparison.json)。

本次观测耗时 252.769 秒，device buffer 分配 30,719 次，host 0，重试 0。整卡采样峰值 2551 MiB，主机最低可用 15,263,887,360 B，cgroup 峰值 11,788,591,104 B；max/OOM/OOM-kill 均为 0。约束保持 16 GiB cgroup、swap 0、3 GiB 可用 RAM、6144 MiB 整卡和 1800 秒，未因结果调整。cgroup 包括文件页缓存，不是 RSS；本例不验证 FP16 host spill、真实显存耗尽或完整模型故障恢复，也不构成性能验收。

四例的源码、实际命令、资源与数值由[独立回读](../execution-audit.json)再次核对；执行有效和官方数值通过分别记录。最终同源码三平台结果见 [CI 归档](../v3/final-ci/summary.json)。所有模型任务已经结束，不重启已完成输出目录。

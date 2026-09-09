# 最终 BF16 兼容路径验证

本轮只执行受 Gemm 修正影响的 BF16 完整模型，不重跑已经通过的 FP32 两例。源码 `7296bfeba2c809d54a7d6214b0f08dee861ad1f8`，ncnn `3b7bdba7`，358 个源码文件、158 项绑定及 5 个实际派生编译单元已在启动前核对。

- 二进制 SHA256：`3202547804114b542a2785a1466bfc117e5d75830bf0633daf7b58cb741910da`。
- 计划 SHA256：`82c096eacb80658a3bd2637218708b1a5ae0381e067823311160cd3d860401c3`。
- 完整快照、模型绑定和原始张量/PNG：`/var/tmp/ernie-memory-execution-20260909-v3`；本目录保留计划、进程、资源与比较的小型记录。
- 配置沿用同一保存的初始噪声与官方参考、15-token apple、512×512、8 步、原生 Vector 文本编码、Vulkan DiT、CPU direct VAE、2 CPU 线程、trace 开启。host 权重，cache/prefetch 0，GPU reserve 512 MiB，buffer auto、spill 2048 MiB、最多 3 次重试。

## 实际结果

完整执行结束，退出 0，所有 8 步与 PNG 已生成，**零 Vulkan validation/VUID**。同一官方参考和原数值门槛下，**17/25 个张量通过，PNG MAE 1.294207255045573、最大差 143（0..255）**；PNG 门槛仍未通过。4,981,760 个值均有限，PNG 与 decoded 的量化精确一致。详见 [full-comparison.json](full-comparison.json)。

这修复了非法 cooperative accumulator 的使用，没有让 BF16 达到完整数值验收。先前升级基线 18/25、PNG max 98 和 v2 的 19/25、max 117 分别属于不同路径；v2 还有 5 条 VUID，保持无效运行记录。当前与旧 BF16 基线仅 6/25 张量逐位一致，不能沿用旧路径的数值结果。BF16 继续作为显式实验选项，默认精度、模型数学与全部门槛没有更改。

本次观测耗时 215.441 秒，device buffer 分配 30,719 次，host 0，重试 0。整卡采样峰值 2262 MiB，主机最低可用 15,278,923,776 B，cgroup 峰值 8,392,908,800 B，max/OOM/OOM-kill 均为 0。保持 16 GiB cgroup、swap 0、3 GiB 可用 RAM、6144 MiB 整卡和 1800 秒的既有约束；cgroup 包含文件页缓存，不是 RSS。单次诊断不构成性能验收，也不是 BF16 host spill 或完整模型故障恢复实验。

本机同源全套 62/62 小型 CTest、0 跳过、零校验错误，包含 4 例 Gemm cooperative ON/OFF 与独立 FP64/BF16 RNE 精确参考，见 [小型前后对照](../bf16-gemm/README.md)。最终源码的三平台构建与小型测试对应 [CI 34373124004](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34373124004)，五个作业全部成功：[独立统计](final-ci/summary.json)为两个 Linux CPU 配置各 37 通过，Linux Vulkan/macOS 各 57 通过、5 BF16 能力跳过，Windows 37 通过、25 无驱动跳过，零失败；各作业另有 23 项 HTTP/清单测试通过。Windows/macOS 完整模型、macOS 生产 RAM 余量查询和物理显存耗尽仍不在本次实测范围。

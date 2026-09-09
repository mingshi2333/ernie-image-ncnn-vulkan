# 后续修正与最终验证

池复用修正按实际 VkDeviceMemory/VkBuffer 记录活动区间，只在连续空闲范围足够时绕过新增 backing memory 的预算检查。失败清理同步的任何非成功状态均 fatal，包括 host/device OOM；预取线程错误也计入 skipped。

本机新官方校验层1.4.357下全量61/61通过；随后局部预取失败计数修正的6项受影响测试通过，均无跳过或validation错误。失败清理注入先真正完成GPU工作，再返回受控错误；没有故意耗尽或破坏设备。

first-ci保存f9dbcf2的全部五个原始CI artifacts与作业身份：Windows37pass24skip，macOS57pass4skip，两个LinuxCPU各37pass；LinuxVulkan38pass19validation失败4BF16skip。它的旧1.3.275校验层无法识别支持的subgroup rotate结构。工作流固定官方LinuxSDK1.4.357.1的校验层/诊断工具及SHA256，系统loader/Mesa不变。sdk-*为实际下载、散列、原始官方发布条目及静态检查证据，sdk-info为本机实际枚举。

## 完整模型：两个 FP32 配置通过，BF16 校验失败

源码 `5b57e9dee4283c5c24fcfdb4bb03349017424253`，二进制 SHA256 `8d8c3216b0d1addf401eeab3d45eddb4aa64b973a05cbf987f20cdb49da49ecd`；357 个源码文件、158 项绑定、4 个实际派生编译单元在启动前核对。计划 SHA256 为 `b794844d3877532e9d9d8eecd813503f089e883be7bf35425b544c9d38e323f3`。完整输入与张量留在 `/var/tmp/ernie-memory-execution-20260909-v2`，小型原始记录保存在本目录。

三例沿用同一保存的初始 latent、15-token 英文 apple 提示词、shared schema-3 模型、512×512、8 步、原生 Vector 文本编码、Vulkan DiT、CPU direct VAE、2 CPU 线程及 trace。保持 16 GiB cgroup、swap 0、主机可用 RAM 至少 3 GiB、整卡采样上限 6144 MiB、每例 1800 秒，没有放宽原始数值门槛。

| 配置 | 完整张量与图像 | 实际内存/预取 | 观测耗时 |
|---|---|---|---:|
| normal-fp32：GPU reserve 512 MiB，cache/prefetch 0 | 25/25 张量及 PNG 与旧 FP32 基线逐位相同；官方 25/25，PNG MAE 0.000361124674479、max 1 | 43,943 次 device 分配，host 0，重试 0 | 258.230 秒 |
| mixed-prefetch-fp32：GPU reserve 5400 MiB，cache/prefetch 各 1024 MiB | 同上，且与 normal-fp32 全部逐位相同 | 43,655 次 device、288 次实际非 device-local host 分配；host 峰值 192 MiB；预取启动/使用 280/280，跳过 0，缓存命中 7，重试 0 | 420.948 秒 |
| bf16-flash-bf16：SDPA 修正后，Gemm 修正前 | 程序完成，但有 5 条非法 BF16 accumulator VUID，不能算有效 Vulkan 通过；原门槛 19/25，PNG MAE 0.905946096、max 117，保留失败 | 完整日志、张量及图像保留；后续单独修正 Gemm 并重新冻结 BF16 | 不用于有效性能结论 |

两例 FP32 分别重算 4,981,760 个有限值；PNG 与 decoded 的量化结果也精确一致。完整指标见 [full-comparison.json](full-comparison.json)，BF16 失败见 [bf16-validation-failure.json](bf16-validation-failure.json)。所有三个进程已结束；原批次保留 `validation_failed`，不因两个 FP32 通过而改写整个批次的状态。

normal/mixed 的整卡采样峰值为 2506/2394 MiB，cgroup 峰值为 8,274,980,864/9,814,110,208 B，后者含文件页缓存，不是 RSS。两例的 max/OOM/OOM-kill 均为 0。mixed 的预取计账峰值为 947,946,496 B，主机时钟观察到的准备与执行重叠为 294.178 秒；这不是 GPU 并行时长或节省的时间。

混合配置确实同时使用 GPU 与 RAM，而且这次总耗时更长。两例同时改变显存预留、缓存和预取，文件页缓存及桌面负载未控制，不能将差异归因于单个选项，也不能承诺加速。GPU reserve 5400 MiB 是受控策略触发条件，没有真实耗尽整卡；本例未触发 OOM 恢复，恢复证据仍来自真实微型图的故障注入。

最初冻结收集器使用错误的 SDPA 文件名，只找到 3 个派生文件，在执行前停止。修正收集器后重新冻结并核对全部 158 项绑定，才开始模型运行；`freeze-preparation-failure.json` 保留原准备失败，不能与已完成的模型运行混淆。

## 平台与剩余边界

源码 `5b57e9d` 的 [CI 34369154353](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34369154353) 已完成，五个原生作业全部成功。各平台通过/跳过数、SDK 身份与原始日志见 [second-ci/summary.json](second-ci/summary.json)。这是 Gemm 后续修正前的源码；修正后的测试结果另行记录，不混用版本。

恢复仅覆盖保留错误类型的 buffer/command 等路径；ncnn 创建 compute pipeline 若将 OOM 折成通用 `-1` 仍会直接结束。全 RAM 512 完整出图、物理显存耗尽，以及 Windows/macOS 完整模型恢复尚无本次验证证据。首轮失败和全 RAM 两步部分结果仍在上级目录保留。

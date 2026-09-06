# 实际 Linux 离线图生图：固定 1024×1024 / strength=.5

新安装包已在完整项目源码隐藏、网络隔离、清除开发 Python 环境、模型只读重绑到中文空格路径后，完成真实原生图生图。最终 PNG 与此前冻结的 native 图像完整 **2,891,207 bytes 逐字节一致**，SHA `d426ac2a82574ed330d1c0f56061f7a5aa335fc2fb87c5bf31083204a50c9e16`。没有修改原输入、原 PNG 门槛或冻结 checker。

本次状态为 `passed_fixed_development_case`。这一个 development fixture 证明新安装包的相应离线使用链可执行；正式质量、速度/内存对比、其他平台和可再分发状态仍未关闭。

| 固定内容 | 实际身份或范围 |
|---|---|
| runtime archive | `d5810af124bac9ebc8b02874e03dfb2a2dd0f5f8b1ede6bb807eb1d2265582d1`，181 项载荷 |
| 安装 CLI | `d366c835fe8d8a7de06db804cae094a4b81b1c64ccdbb005551d8942aee7c267`，来自已记录 Clang Vulkan 构建 |
| checker | `4f959a945ff7496f03084b59832ae47fda0e7b8235c0a1258667ce47a7de57f3`，与旧 v2 失败检查相同 |
| 执行 | native Vector CPU text、FP32 Vulkan DiT、CPU direct VAE，steps8 的 strength=.5 后缀4步；trace关闭 |
| 输入 | 原1024图像、带LF prompt及保存FP32 noise；完整来源均在 case.json |
| 资源限制 | 独立 systemd scope，10GiB memory.max、swap.max0、CPU200%、CPU8/10；host available至少3GiB，每命令1800秒 |

九个实际命令均按原合同结束：isolation、help、diagnose、verify-model、无驱动 diagnose、完整生成、生成后 verify-model 返回0；缺模型和已存在输出返回1。现有输出仍保留原SHA。无驱动诊断现在返回零设备及 `vulkan_error`，相应程序修复及先前 v2 失败见 `../offline-linux-diagnostics/`。

生成前后原生 CLI 都完整验证模型。全部188项准备/输入/runtime SHA前后匹配；原工作区在 namespace 中为空tmpfs，清理环境与不同网络namespace有实际证明。系统动态库和GPU驱动仍来自本机，这不是最小rootfs、跨发行版ABI或Windows/macOS验证。

## 资源观察

| 阶段 | 父进程记录时间 | 同一supervisor cgroup采样峰值 | 事件 |
|---|---:|---:|---|
| generate | 505.09 s | 6,489,657,344 B | max/oom均0；host最低10,233,769,984 B |
| verify-model-after | 23.43 s | 10,737,418,240 B | 触及10GiB限额，累计max49634，OOM各项0；命令正常退出 |

外层会话记录总时间549.33秒，包含完整验证和诊断。`memory.current` 是整个supervisor cgroup的采样值，不是进程RSS或精确GPU分配器峰值；没有保存各组成的memory.stat，因此不把末次压力归因于某一缓存或RSS项。原checker并未把max事件非零设为失败门槛，本次如实保留该观察，没有看到结果后更改验收。不得把生成阶段6.49GB写成完整链峰值，也不得声称全流程max0。

独立审查 `0fb6120 / 8131e27` 重建九条命令隔离参数、绑定载荷/源码/输入、重算完整PNG及CRC，确认原固定case通过且资源压力记录准确。`manifest.json` 绑定本目录的原始证据，不包含本说明本身。图像和压缩包不放入Git，实际保留路径及SHA在 `large-artifacts.json`。

可发行资格保持 `distributable=false / published=false`。本次尚未执行完整D3的长中文、PE、只读输出和三次重复等场景；Windows、macOS、公开下载与人工质量评价仍需独立证据。

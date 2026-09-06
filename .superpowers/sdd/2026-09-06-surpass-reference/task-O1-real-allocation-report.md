# O1 第三切片：真实 allocator 合同与完整链接准备

## 当前状态

新增真实 Vulkan 合同 `tests/test_allocation_metrics_vulkan.cpp`，独立目标 `ernie-allocation-vulkan-contract`，由默认关闭的 `ERNIE_BUILD_ALLOCATION_PROBE` 启用。**ON/OFF 完整链接、真实allocator合同与1024-byte GPU输出逐位比较已通过。只完成受测小合同，不代表完整模型性能指标接入或全O1完成。**

独立目录 `build-o1/on` 与 `build-o1/off`；原始 ncnn 仍为 pin `6a1bf000f363714839a36793addc8c879d3d899e`。ON 使用认证派生副本，OFF 直接原始源。Release、`/usr/bin/clang{,++}`、system glslang 16.2.0 与 build-dev 配置匹配，保留项目当前 minimal layer inventory。没有改写/配置/构建 build-dev，也没有加载正式模型。

## 合同实际覆盖设计

- Session 在 `create_gpu_instance` 之前启动；所有用户 Net、Mat、allocator、command 析构后销毁 GPU instance，再取得最终快照，检查实际内存 live=0、所有 allocator inactive 且无 live handles。
- Blob 的 4096-byte 请求使用 1 MiB pool：实际分配后计数增加，fastFree 不能减少物理 live，再申请必须复用相同 memory+offset、live 不变，clear 后必须恢复之前的物理 live。
- 小 image allocation、单独 cache-tagged blob allocator、普通与 host-preferred weight allocator、staging allocator，均真实申请、归还和清理。ON 最终必须观察 Weight/Blob/Staging/Cache 四角色的成功实际分配。host-preferred 不保证 host import；以真实事件分类为准。
- 原始 pinned BinaryOp GPU 执行 256 个确定的 FP32 值乘二，实际上传/计算/下载后逐元素检查精确 oracle，并保存 1024 bytes 原始 FP32 输出。ON/OFF 使用相同源码和输入，随后对输出作逐位比较。
- JSONL 保存各检查点、实际 live/peak/count、device/memoryType/heap/property/import 分类和 allocator address/generation/role/active 状态。OFF 字段明确 unavailable/null，默认 OFF 目标不包含也不链接 observer。
- 无 GPU 返回 77，参数错误 2，任何申请/计算/计数/清理/文件关闭错误返回 1。失败记录不能作为通过证据。没有应用模型精度改动。

## 已执行验证与资源状态

证据目录 `outputs/allocation-real-o1-v1`。

- ON 配置成功，峰 179.7 MiB；OFF 配置成功，峰 43.6 MiB。
- ON/OFF 两次 `clang++ -fsyntax-only -Wall -Wextra -Werror` 均成功，未执行 GPU。
- ON 完整构建使用 systemd user unit `ernie-o1-build-on`，MemoryMax=3221225472、MemorySwapMax=0、并行度 2；统一执行会话 59085。
- 收到 root F1 PE 加载预约后，执行 freeze 并核验 FreezerState=frozen。暂停时 87%，MemoryCurrent=321421312 B、MemoryPeak=329334784 B。构建中间文件与会话保留。OFF 未启动编译。
- 日志 `configure-on.log`、`configure-off.log`、`build-on.log`。等待 root 允许后 thaw 同一 unit，禁止因为观察超时重启。

## 后续执行

继续 ON 后仅构建同一 target；随后在相同 3 GiB/swap0 资源合同下构建 OFF。二进制固定复制到新 evidence snapshot，保存完整 source/CMake/derived provenance/runner SHA。实际 GPU 必须等 root 独占时段，再串行执行 ON、OFF，同输入/output-byte oracle、全部事件检查通过后比较 SHA 与 bytes。

本合同仅证明受测 allocator 和小 GPU 运算的事件覆盖与插桩透明性；不能将其外推为完整模型性能证据、driver 总显存或任意进程的峰值。完整端到端指标胶合与真实模型覆盖仍由后续 O1 切片完成。

## 完整链接续接与身份封存

root 通知 F1 已完成 PE/text、host available 17 GiB 后，thaw 原 `ernie-o1-build-on`，没有重新启动。ON session 59085 exit0，CPU 3m6.129s、峰 597.4 MiB/swap0；wall 9m42.810s 包含冻结时间，不能作编译性能比较。OFF 在独立 unit `ernie-o1-build-off`、session 53153 中完成，exit0，CPU 3m25.844s、wall 1m44.323s、峰 599.7 MiB/swap0。均为2 jobs/3 GiB限制。

固定 runner：

- `outputs/allocation-real-o1-v1/snapshot/on/runner`，SHA256 `e2d0e552abcdcecba89886684d2fbb8ddd6c35ef92a3e421eb62a09a1b1906b4`。
- `outputs/allocation-real-o1-v1/snapshot/off/runner`，SHA256 `fb82ee70125e3aacbae916f8aafafbba41b7e2979149c4eefbbc4da092e3c464`。

`identity.json` 保存 ON/OFF 实际 `.o.d` 中1202/1168项编译依赖SHA（项目/ncnn/生成源复制封存，系统header记录SHA），实际 linker dependency 文件中的31/29项链接依赖SHA、CMakeCache/flags/link命令、派生源码完整provenance/差异、配置脚本快照和固定输入文件SHA。`seal.py` 与其SHA一起保存。两runner均只执行无参数用法路径并确认exit2，不创建GPU。原始 ncnn checkout仍clean。

真实GPU执行继续等待root F1/Q2独占队列。此时无本agent活动CPU构建/模型会话。

## 真实GPU执行：保留v1失败，v2通过

root明确放行后各运行封存v1 ON/OFF一次，均在约1s处因GPU输出oracle断言失败。真实pool分配/归还/复用/clear与各角色分配事件此前均valid；失败发生于probe图的标量参数。v1的 `2=2` 被pinned ncnn ParamDict存成int位模式，BinaryOp按float读取；应为 `2=2.0`。这不是observer改动数值的证据。v1未到最终快照，不将其当作清理完成证明。原runner、源码、两份失败日志/events保持不变。

新增CPU同图preflight，在创建GPU前验证参数与256值精确oracle，并提供 `--cpu-preflight` 纯CPU入口。两份新runner在CPU通过后封存到 `outputs/allocation-real-o1-v2/snapshot/{on,off}/runner`。独立只读ParamDict证明：`2=2` 返回float `2.8026e-45`、bits `0x2`；`2=2.0` 返回float2、bits `0x40000000`。证明最初直接调用protected方法编译失败、之后缺glslang静态链接依赖失败，均保留日志；最终测试派生类公开原方法、链接原ncnn及同一系统库，CPU证明通过。没有修改ncnn或真实模型数值代码。

root再次明确放行后v2 ON/OFF各实际GPU运行一次，均exit0：

| 项目 | 结果 |
|---|---|
| ON runner SHA256 | `c8df6e7d20e783ece47cf1ba9f4d729816a4b3bfaf0efc2d7a57dc666eb91d28` |
| OFF runner SHA256 | `b9f4d9c39c895d08a23433f555a6a6fc549df107ba3bb96dad8282027b0aabf0` |
| 输出 | 1024 bytes逐位一致，SHA256 `990fabcba00d265c85b37a1d1a73d18c1c0abb87d944584f70000bf2ab62544c` |
| ON总物理allocation次数/同时峰值 | 11 / 2,134,016 bytes |
| 最终清理 | destroy_gpu_instance后live=0，所有allocator inactive且无live handle |
| 角色 | Weight/Blob/Staging/Cache实际分配均观察到 |
| host import | 实际观察到1次，1,048,576 bytes；独立host heap/type分类 |
| 设备本地/映射 | 实际内存type1 property1、type4 property7在heap0；host import type2/property6与staging type3/property14在heap1 |
| OFF统计 | 所有检查点available=false，live/peak/count=null |
| 资源 | ON979ms/OFF921ms；两者cgroup峰157.1MiB、swap0，未加载模型 |

所有ON检查点valid。blob小请求归还后live未减，二次申请复用同memory+offset且无新物理计数，clear精确释放。最终类级live均为0。**各类独立峰值不能求和当同时峰值；2,134,016是事件时间线直接统计的总同时峰。**host-import属于VkDeviceMemory，但不能统称独立显卡VRAM。覆盖仍为受观察ncnn allocator，不包含driver内部所有malloc/其他进程分配。

`result.json`绑定本次immutable `identity.json` 的SHA、输出/events/logs/ParamDict证明SHA。identity中的execution pending是执行前封存状态，执行结果在独立result中续接，未改写历史身份。GPU完成后立即通知root释放；无活动GPU进程。本结果不是完整模型速度、显存峰值或正式S/M测量。

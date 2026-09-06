# O1 第三切片：真实 allocator 合同与完整链接准备

## 当前状态

新增真实 Vulkan 合同 `tests/test_allocation_metrics_vulkan.cpp`，独立目标 `ernie-allocation-vulkan-contract`，由默认关闭的 `ERNIE_BUILD_ALLOCATION_PROBE` 启用。**当前仅 ON/OFF 严格编译语法检查与 CMake 配置成功；ON 完整 ncnn 构建在 87% 由 root 的 F1 资源排队指令冻结，真实 GPU 合同和 ON/OFF 逐位比较均未执行。不得称 O1 完成。**

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

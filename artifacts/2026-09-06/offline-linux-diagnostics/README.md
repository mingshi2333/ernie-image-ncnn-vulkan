# Linux 无驱动诊断修复与安装回归

实际离线 v2 检查先通过 namespace、help、正常设备诊断和完整模型校验，随后因无驱动 `--diagnose` 返回 1 而停止，未进入生成。原 `vkCreateInstance failed -9 / Cannot initialize Vulkan`、命令、资源状态及失败结果完整保留于 `prior-offline-failure/`，没有用新版结果覆盖。

修复使可选诊断在 Vulkan 初始化失败后清理实例、不登记会话，返回 `vulkan_compiled=true`、`gpu_count=0`、`default_gpu_index=-1` 和 `vulkan_error=Cannot initialize Vulkan`。它跳过会隐式重新创建实例的 ncnn 查询；强制 Vulkan 推理仍抛错，CPU 路径不需要可用 Vulkan。已有嵌套、并发及外部借用实例行为保留。公共 `DiagnosticInfo` 追加标准库字符串，支持旧源码重编译，未声明旧二进制 ABI 兼容。

固定源码 `3a112e7b9df9bb5be02ff21529773966a596d6563dfb694bf50892cec9fa3a27` 包含 288 项文件及明确的 8 项覆盖。它相对旧 D1 Clang v2 的生产文件只改变 CLI 打印、公共诊断字段、GPU context 和 runtime_info，所有模型数学源码相同。

| 实测 | 结果 |
|---|---|
| 全新 Clang Vulkan 配置 / 构建 | 成功；构建 149.39 秒 |
| Vulkan 受影响的 CTest | 6/6 通过，无跳过；包含真实无驱动环境和正常 GPU context |
| 全新 Clang CPU 配置 / 构建 | 成功；构建 127.06 秒 |
| CPU 受影响的 CTest | 5/5 通过，无跳过 |
| 新 Vulkan 安装前缀移动、源码隔离、网络隔离 | 外部 C++ 配置、链接、运行，以及 CLI help/diagnose 成功；损坏包按预期拒绝 |
| 新安装文件 | 75 项，安装 CLI SHA `d366c835fe8d8a7de06db804cae094a4b81b1c64ccdbb005551d8942aee7c267` |

两次构建各使用实际 4 GiB、swap0、CPU200% 和 CPU8/10 限额；源码前后完整散列相同，固定 ncnn 无 tracked 修改，OOM/max 事件为零。构建耗时属于资源记录，不是推理性能比较。

独立审查 `55e58ef / 5de7279` 验证 Vulkan 源码、二进制、日志和全部 75 项安装文件；后续 CPU 结果保存在本目录。`manifest.json` 绑定原始证据，不包含本说明本身。新的 full-model 离线运行仍在单独执行，不能以这些小合同证明生成、Windows/macOS 或可分发状态。

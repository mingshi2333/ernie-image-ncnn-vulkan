# D3 无驱动诊断窄修复独立审查

结论：本切片无剩余阻断。只读审查 root 的八文件变更，并认证新冻结源实际构建与六项针对 CTest；未运行完整模型，不宣称 D3 离线生成已经通过。原 `/tmp/ernie-offline-img2img-1024-v2` 无驱动 diagnose 退出 1 的失败仍保留，不能把本次小合同成功覆盖为该次生成成功。

## 生命周期与接口

- `GpuContext` 首个 owned 初始化失败时先调用 `destroy_gpu_instance()`，清 `context_owned`。`require_device=false` 保存错误并返回，未增加 `context_users`、未置 `enabled_`；析构不会减一个未获得的引用。`require_device=true` 继续抛 `Cannot initialize Vulkan`。
- pinned ncnn `6a1bf000f363714839a36793addc8c879d3d899e` 的 `get_gpu_instance()` 为非创建查询，但 `get_gpu_count()` / default-index 查询会经 `try_create_gpu_instance()` 自动初始化。`runtime_info.cpp` 新增 `available()` 分支明确避免失败后再次创建或读取旧计数。成功初始化但零设备仍报告 `No Vulkan device`。无驱动时 model.cfg 元数据继续读取，错误配置依旧不被吞掉。
- 正常 nested / concurrent / borrowed 路径以及异常 index 的清理路径未修改。重复 missing-driver 测试验证两轮 diagnose 后 strict 构造仍拒绝且 instance 保持 null；原上下文合同继续实际通过。
- `DiagnosticInfo` 只追加 stdlib `std::string` 字段，没有引入 ncnn 或 PNG 头依赖。普通源码消费与尾部省略 aggregate 初始化兼容；这是源码兼容边界，结构体布局变化不保证已编译消费者 ABI 兼容，消费者应以匹配的新头文件与静态库重建。
- CLI 仅为 diagnose 增加可选 `vulkan_error` 文本字段；强制 Vulkan 生成仍 exit 1 且不写 PNG。修复不触及模型数学。

## 实际证据核验

冻结 `outputs/d3-runtime-source-v3/source-identity.json` SHA256：
`3a112e7b9df9bb5be02ff21529773966a596d6563dfb694bf50892cec9fa3a27`。
基线 `73f1c1e0088d7dd5633bed8d408a703b632aba86`，288 文件、仅声明八项 overlay。本审查逐文件重算全部 288 SHA，八项与当前审查源相同；重算三份 configure/build/test 日志和 33 个构建二进制，均匹配 `outputs/d3-runtime-build-vulkan-v3/{identity,result}.json`。构建结果记录 source_changes_after_build 空、ncnn_changes_after 空。

实际全新 Clang 22.1.8 Linux Vulkan preset：configure/build/test 各 exit 0，build 149.390 s；six CTests 总 3.62 s，6/6 passed，无 skip：pipeline_api_contract、request_validation_contract、gpu_context_contract、gpu_context_missing_driver、cli_contract、installed_cpp_consumer。本审查读取并认证这些既有实际测试，不另开 GPU 工作。

实际 CLI SHA256 `d366c835fe8d8a7de06db804cae094a4b81b1c64ccdbb005551d8942aee7c267`；GPU context contract SHA256 `c085cf23f83696fa09bb809c1b328631a77b0c522c07133a4c5cc1bdc91c26c7`。构建证据声明 4 GiB memory.max、swap.max=0、CPU 8/10、2 CPU quota，memory.events 的 OOM / OOM-kill 为 0；不是推理峰值测量。

本结论限缺驱动错误恢复及接口小合同。新安装包的迁移隔离、实际 ICD 装载和离线完整生成还需独立新运行证据；不外推 Windows/macOS、完整质量或性能验收。

## 新安装消费补充

随后完成的 `/tmp/ernie-d3-vulkan-install-v3/result.json` 状态 passed。本审查另逐文件重算移动后安装前缀全部 75 文件 size/SHA，全部匹配。实际命令包含移动前缀、bwrap 整个项目 tmpfs 隐藏、unshare-net、外部消费者 configure/build/test 和 CLI help/diagnose，均 exit 0；损坏包实际 exit 1 且输出精确 `Unexpected schema-3 fields`。这次 installed diagnose 枚举三设备（NVIDIA 与两份 llvmpipe），不是缺驱动分支；缺驱动由前述定向 CTest 单独验证。这里只增加真实隔离安装消费证据，完整离线模型生成仍待执行。

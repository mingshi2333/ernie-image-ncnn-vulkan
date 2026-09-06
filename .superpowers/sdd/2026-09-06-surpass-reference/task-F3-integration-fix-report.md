# F3 图像错误和 GPU 生命周期修复

四个独立复现的 Important 已修复，并经 84b9bcb 独立复审关闭。此结论仅覆盖本切片，完整 F3 功能/跨平台验收仍未关闭。

- stb 错误字符串为空时给出稳定异常，不再对空指针拼接字符串。decoded 采用 RAII，分配/拷贝异常也释放图像。
- BMP/TGA/JPEG 通过有界 callbacks 解码；首次合法短预取可用，实际越过 EOF 或越界 skip 被拒绝。编码回调的内存/大小失败显式传播。
- 输出采用平台原生独占创建，处理短写和 EINTR 并检查 close。已有文件不会被竞态覆盖；真实写入失败会返回错误。失败的新路径可能保留部分数据，未宣称自动清理。
- generation 与 diagnose 共享线程安全 GPU 使用计数。最后一个库内使用者才释放本库创建的实例；外部已存在实例被借用，其所有者须保持生命周期。本组件保护实例生命周期，完整模型依旧按项目要求串行。

构建：当前 build-dev Release 的 ernie-image、ernie-image-io-contract、ernie-gpu-context-contract、ernie-request-validation-contract 全部成功。第一次构建开始时新测试 target 尚未写入生成文件，出现 No rule to make target；刷新配置后的 session 12061 完整 exit 0，保留该过程而不记作产品错误。

聚焦 CTest：request_validation_contract / gpu_context_contract / image_io_contract / cli_contract，4/4 通过，用时 6.15 秒。
GPU 测试在当前机器真实枚举并检查嵌套、跨线程并发诊断、无效子请求、最后所有者释放、外部实例借用；没有运行 GPU 模型计算。图像测试包含当前编解码器创建的合法图片往返、有效头截断、JPEG 尾截断、既有输出未改变，以及子进程 RLIMIT_FSIZE=0 的实际写失败。独立评审另行重编旧复现程序，全部改为正常拒绝，无 SIGSEGV。

Windows 专用独占写分支已有实现，真实 Windows 运行仍属于 P4/P5，未由 Linux 测试代替。

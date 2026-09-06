# D2 下载契约独立复核

审查对象：初始 `67cdafb`，路径修复 `1f1d444`。root 只读检查下载/清单实现，实际重跑标准库 localhost HTTP 和文件系统测试，未下载外部模型或执行 Windows。

初版 20/20 通过，但独立负例确认 `weights/LOCK.PART`、`.DOWNLOAD.LOCK`、`weights/a?.bin`、`weights/a|b.bin` 被接受。Windows 大小写不敏感的内部恢复/锁名称，以及禁止文件名字符没有得到完整保护。此处是输入契约缺口，不是已有 Windows 运行结果。

`1f1d444` 对内部后缀和锁名使用 casefold，并拒绝 Windows 特殊字符、设备别名与带扩展名的别名。补充两个测试后，root 实际重跑 `python3 -W error::ResourceWarning -m unittest tests.test_model_download tests.test_release_manifest -v`，22/22 通过；上面四个独立负例均被拒绝。保留原始问题记录，不把跨平台路径规则的 Linux 测试称为 Windows 实机通过。

实现确实流式读取，完整 SHA 认证后才创建最终文件；206、200 重取、版本变化、截断、尾字节不匹配和原子发布不覆盖均有实际 localhost/文件系统行为测试。现有目标不匹配时保留原字节；完整目标必须认证后才能跳过。目标文件系统需要硬链接，容量预检保守，强制终止后的残留锁需要确认旧进程退出后处理，这些边界已在作者报告中列明。可信本地目录是前提，没有承诺抵御其他本地进程持续替换目录。

本次下载切片无未关闭的 Important finding。结果仍只是下载字节完整性；原生包语义校验、规范化全权重审计、真实 Windows 下载、许可依据、公开资产与离线实际生成不由这 22 项测试证明。

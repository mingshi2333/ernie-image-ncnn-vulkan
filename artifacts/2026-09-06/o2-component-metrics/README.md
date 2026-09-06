# 保留的 O2 构建启动失败

这份早期构建尝试在连接用户服务总线时退出：缺少 `DBUS_SESSION_BUS_ADDRESS` 和 `XDG_RUNTIME_DIR`，记录的 exit code 为 1。日志没有编译器或模型执行结果，不能视为数值测试失败。后来使用显式用户总线环境的编译和实际验证已完成，见 [O2 实际结果](../../2026-09-07/component-loading-regression/README.md)。本目录保留原始失败日志，不覆盖为成功。

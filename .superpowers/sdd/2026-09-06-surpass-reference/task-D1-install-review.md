# D1 安装接口独立复核

审查对象为 2026-09-06 root 尚未提交的 D1 切片；精确源 SHA 见 `/tmp/ernie-d1-independent-review-v1/review-identity.json`。本审查只读生产源码，没有构建或运行 GPU/模型；CPU 小配置负例的完整输出保存在同目录。未修改原安装前缀或上游。

## 结论与 Important finding

Linux CPU 的移动安装、公共 C++ 消费和 CLI 模型前拒绝证据有效。但固定依赖身份仍有一项可复现缺口，在修复前不能声称安装接口始终使用同前缀 pinned ncnn。

**Important：调用方 `ncnn_DIR` 可以覆盖同前缀依赖。** `cmake/ErnieConfig.cmake.in` 第 16 行的 `find_dependency(ncnn CONFIG PATHS ... NO_DEFAULT_PATH)` 仍优先读取 CMake 的 `ncnn_DIR` 缓存。包前面的 `if(TARGET ncnn)` 只能拒绝已导入 target，无法拒绝尚未加载的外部包路径。本审查新建只有 `add_library(ncnn INTERFACE IMPORTED)` 的外部 `ncnnConfig.cmake`，传 `-Dncnn_DIR=/tmp/ernie-d1-independent-review-v1/fake`、同时指定真实移动安装 prefix，`find_package(Ernie CONFIG REQUIRED)` 配置成功并打印 `REVIEW_FOREIGN_NCNN_LOADED`，退出 0。完整命令与输出可从 `build/CMakeCache.txt`、`foreign-ncnn.log` 复现。真实其他 ncnn 版本可以因此改变链接对象、ABI、层清单或数学，当前 `Ernie_NCNN_REVISION` 仍无条件声称固定版本。

建议强制选择安装目录内确定的 ncnn config，避免调用方缓存/查找重定向覆盖，并补外部 `ncnn_DIR` 回归；保留对已导入 ncnn 的拒绝。此处只报告，未代作者修改。

## 实际证据与失败处理

- `/tmp/ernie-sdk-cpu-install-v1/result.json` 为 passed；重新逐项核对 51 个安装文件的完整 SHA256/大小全部匹配。源/构建记录位于 `outputs/d1-install-build-cpu-v1`；这是一次具体 Linux CPU 配置的证据，不是所有配置的保证。
- 安装目录确实从包含空格/中文的 original 路径移动到 `移动 installation`；外部消费者的 CMake 编译命令只获得该前缀的公共 `include`，没有私有 src/ncnn include 传播。公共 header 使用 stdlib；底层 static link-only 实现依赖仍正确保留。安装的 detail targets 可被用户主动引用，但正常 `ernie::pipeline` 消费不泄露私有头文件路径。
- bwrap 的实际命令隐藏整个 `/home/mingshi/Project/AI/ernie-image-ncnn-vulkan`（含共享 ncnn、工作树和原 build），禁网，再配置/链接/执行消费者及 CLI。允许系统编译器、标准库、OpenMP 和系统设备存在；并非最小 rootfs 或跨系统二进制兼容证明。
- `consumer-test.log` 确实执行 `installed_pipeline_api`，1/1 passed；不是只有 configure/link。消费者检查公共 API 形状错误的特定异常文本与 progress 未调用、缺失模型验证，以及 CPU diagnose 无 Vulkan 设备。
- CLI help/diagnose 的退出 0 与损坏 schema3 包退出 1 均实际存在，后者日志为 `Unexpected schema-3 fields`。启动缺共享库不会被前两个成功检查误判通过；损坏包测试目前只固定退出码，没有对错误文本做自动断言，本次日志已核对。
- 独立构造两份小依赖前缀：分别遗漏 `libernie_tokenizer_bridge.a` 和 `libncnn.a`，其他 archive 只引用原安装文件；配置均退出 1，且明确报告被遗漏路径。日志 `missing-libernie_tokenizer_bridge.a.log`、`missing-libncnn.a.log`。没有通过移除原文件制造干扰。
- unittest 默认需要显式 `ERNIE_INSTALL_BUILD`，否则 skip；一般 unittest 入口也没有 source isolation。本次隔离成立来自实际 `--isolate-source` 命令与 bwrap 记录，不能把普通 skip 或非隔离测试算同等证据。

## pkg-config 与声明边界

实际 pinned 上游 `third_party/ncnn/src/ncnn.pc.in` 与生成/安装 ncnn.pc 均是 `prefix=${pcfiledir}/../..`，不是固定 `/usr/local`，故此固定版本无需本项目重定位覆盖；没有修改上游。Ernie 的被验证消费接口为 CMake target，不声明单独 ncnn pkg-config 静态依赖完整性。

SDK 配置拒绝 instrumentation、共享 ncnn、Vulkan + system glslang 组合，以收窄可迁移静态导出范围。当前只验 Linux CPU。Vulkan bundled glslang 安装链尚需实际隔离消费者验证；Windows/macOS、不同标准/OpenMP runtime、真正图像生成不在这次证据范围。现有测试不是 hermetic 工具链证明，源 inventory/构建记录也不代替未执行平台的运行证据。

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

## v2 实际 CPU/Vulkan 安装闭环

root 将修复冻结在 `outputs/d1-install-source-v2`，base_commit12cd53a加显式12项overlay，source-identity.json SHA70faadbee76bc31f7728c010d1e9371d99a2e2aefa8e7bdc74ef60847645c811。审查重新核对其中94个项目源文件全部hash匹配，两份 `outputs/d1-install-build-{cpu,vulkan}-v2/identity.json` 都绑定此同一源identity。两build result均exit0、source_changes_after_build空。此次审查未重建或运行GPU；底层ncnn/glslang依赖身份仍以固定来源与构建记录为准，不把94个项目文件hash称完整工具链闭包。

实际结果 `/tmp/ernie-sdk-cpu-install-v2/result.json`、`/tmp/ernie-sdk-vulkan-install-v2/result.json` 均passed。重新完整流式SHA/size核对CPU51个、Vulkan75个安装文件，全部一致。两者的恶意 `foreign-ncnn/ncnnConfig.cmake` 实际含FATAL_ERROR；consumer配置传入对应 `-Dncnn_DIR` 后成功，consumer缓存仍为原来的 `ncnn_DIR:UNINITIALIZED=/tmp/.../foreign-ncnn`。安装config现在直接include确定同前缀ncnnConfig，未执行重定向外部包；原Important在源码与真实回归两层关闭。

两套消费者配置/链接/执行均由bwrap隐藏整个项目根（含原构建/ncnn），unshare-net，移动中文空格安装前缀后进行。命令日志退出：install/preflight/configure/build/CTest/help/diagnose均0，损坏包验证1并输出Unexpected schema-3 fields。两份CTest均明确执行installed_pipeline_api，1/1通过，分别0.00秒和0.26秒，非空测试假通过。compile_commands只注入移动前缀public include，没有私有ncnn/src include路径。Vulkan额外安装bundled shader compiler静态依赖后实际完成链接，不依赖被隐藏的原build。

CPU diagnose输出vulkan_compiled=false/gpu_count0；Vulkan diagnose实际完成设备枚举（RTX4060及llvmpipe）。这是API/安装/枚举合同，不能称Vulkan fullmodel推理、速度或质量验收，也没有验证Windows/macOS或不同工具链/驱动的可移植性。既有模型前异常检查与CLI错误处理证据继续成立，当前D1本次Linux CPU/Vulkan安装切片无未关闭Important。

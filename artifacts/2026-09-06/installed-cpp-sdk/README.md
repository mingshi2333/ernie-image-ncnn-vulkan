# Linux 安装后 C++ 接口验证

Linux CPU 和 Vulkan 静态 SDK 均已在移动安装目录、隐藏整个源码仓库和原构建、禁网后供独立 C++ 程序成功使用。公共消费接口为 `find_package(Ernie 0.1.0 EXACT CONFIG REQUIRED)` 和 `ernie::pipeline`。这项证据覆盖安装、链接、API 拒绝行为与设备枚举，没有运行完整 ERNIE 模型，也不代表 Windows/macOS 已验收。

## 实际构建与消费

| 项目 | Linux CPU v2 | Linux Vulkan v2 |
|---|---:|---:|
| 冻结项目源文件 | 同一组 94 个 | 同一组 94 个 |
| 构建退出 | 0 | 0 |
| 构建用时 | 131.75 s | 152.61 s |
| 构建进程树 time 最大 RSS | 690268 KiB | 690964 KiB |
| 安装文件完整身份复核 | 51/51 | 75/75 |
| 外部消费者实际 CTest | 1/1 | 1/1 |
| 隐藏源码/原 build，断网 | 是 | 是 |
| CLI help / diagnose / 损坏包 | 0 / 0 / 1 | 0 / 0 / 1 |

构建使用固定 ncnn `6a1bf000f363714839a36793addc8c879d3d899e`、静态 Rust tokenizer，Vulkan 使用同前缀 bundled glslang 静态库。构建编译器为 Clang 22.1.8。实际构建身份、命令与完整日志在 [build](build/)，原始隔离命令、安装文件 SHA256/大小、外部编译命令和 CLI 日志在 [install](install/)。构建统计只用于资源记录，不作为生成性能对照。

冻结项目源身份 SHA256 为 `70faadbee76bc31f7728c010d1e9371d99a2e2aefa8e7bdc74ef60847645c811`，base commit `12cd53a` 加明确列出的 D1 overlays；两个构建均在前后核对这些项目源字节未变。该清单不是整个编译器、系统库、ncnn/glslang 的完整 hermetic 工具链闭包。

安装测试将前缀从 `安装 original` 移到 `移动 installation`。随后由 bubblewrap 隐藏整个项目根（包括共享 ncnn、worktree、模型与原 build），禁用网络，清除外部库/include 搜索环境，并配置、链接、执行外部消费者。系统编译器、标准/OpenMP runtime 和设备仍存在；此环境不等于最小 rootfs，也不证明跨 Linux 发行版 ABI 兼容。

公共 C++ 源只包含 `ernie/pipeline.h` 和标准库；编译器搜索路径没有私有 ncnn/src，CMake 未导入 PNG/JPEG target。实际调用验证非法宽高在推理与进度前被拒绝、缺模型通过原生 tokenizer/package bridge 被拒绝，随后调用 diagnose。CPU 记录 Vulkan 未编译；Vulkan 记录实际设备枚举。安装包的内部归档保持链接依赖，没有将 CLI 图像文件库加入公共接口。

## 独立审查发现并关闭的问题

v1 的 `find_dependency(ncnn ... NO_DEFAULT_PATH)` 仍会优先读取调用方缓存 `ncnn_DIR`。独立审查用另一份 ncnnConfig 实际复现替换成功；此历史失败没有删除。v2 改为 include 同安装前缀的确定 ncnnConfig，并在真实消费测试里传入含 `FATAL_ERROR` 的外部 ncnnConfig。CPU/Vulkan 都成功配置且调用方 cache 原值不变，确认没有使用外部 ncnn。完整过程见 [独立复核](independent-review.md)。

SDK 导出拒绝 allocation instrumentation、共享 ncnn，以及 Vulkan + system glslang 的未覆盖组合。对已导入的独立 ncnn target 明确拒绝，避免两个不同数学/ABI 依赖混用。独立删除副本中 Rust/ncnn archive 的负例也在配置时失败；原安装文件未被移动或破坏。

## 范围与继续验证

新的 Linux CPU preset v5 已在 GNU 16.2.1 上实际配置、构建并通过完整 **27/27 CTest**，其中包含真正安装、移动前缀和外部消费。267 个冻结项目源文件前后不变；ncnn tracked 源未变，实际 cgroup 为 4 GiB、swap 0、CPU quota 200%，CPU 亲和性为两个物理核。完整记录见 [preset/cpu-v5](preset/cpu-v5)，独立审查另使用同一实际 CPU runner 重跑 CLI/text bucket **34/34，无 skip**，见 [增量复核](preset/independent-review.md)。普通 CTest 的移动前缀测试不自行隐藏源码；上文的独立 bwrap 命令才是隔离证据。

CPU preset 的历史失败也被收录：v3 worker 误对未初始化的 worktree 子模块路径检查 Git，尚未配置便拒绝；v4 更正依赖路径后构建成功，但只有 25/27 CTest。两个老 CLI 断言默认 Vulkan 已编译，CPU-only 实际正确报告 `Built without Vulkan`；另一个 text bucket 用例依赖未纳入该源快照的历史 artifact。修复使用实际 diagnose 能力选择明确错误，纯尺寸测试指定 CPU FP32，并将原 32-token 图的相同字节保存在 tests/fixtures。v5 完整重建和测试通过，没有改变推理数学、跳过失败用例或放宽数值门槛。

同一 267 文件冻结源码上的 GNU 16.2.1 Vulkan preset v5 已实际配置、构建并通过 **42/42 CTest**，没有 skipped；完整 LastTest.log 确认 NVIDIA GeForce RTX 4060 Laptop GPU 上的 cache、attention、FP32/FP16/BF16 小算子执行。配置 6.36 s、构建 281.29 s、CTest 15.76 s；构建和测试独立保存 source/worker 身份，测试又绑定构建身份与全部目标二进制，前后源码与二进制不变。见 [Vulkan preset 实测](preset/vulkan-v5) 与 [独立复核](preset/vulkan-v5/independent-review.md)。这补足 Linux 预设验证，前述 Clang Vulkan 安装 v2 保留为独立隔离消费证据。固定 Rust 1.98.0 与 SDK/下载合同已接入本地 CI 配置，远程 Actions 尚未执行。Windows/MSVC、macOS/MoltenVK、实际完整模型离线安装链、第三方 notice 闭包和公开下载均需各自证据。当前 SDK 0.1.0 是源接口，不承诺跨工具链稳定二进制 ABI。

原始 CPU v1 与 Vulkan v1 构建/安装记录在本地 `outputs/d1-install-*` 和 `/tmp/ernie-sdk-cpu-install-v1` 保留；本 artifact 以受审查 v2 为主。后续 preset 的成功与失败单独记录，不回写这些构建或安装结果。

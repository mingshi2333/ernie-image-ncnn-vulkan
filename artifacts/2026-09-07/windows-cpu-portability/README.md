# Windows CPU 交叉构建、UTF-8 与迁移安装

本轮修复了实际 MinGW/Wine 检查发现的构建、文件覆盖和 Unicode 路径问题。最终 Windows CPU 程序完成 28 项 C++ 检查、35 项命令行/报告检查及迁移安装检查；Linux CPU 完整 CTest 35 项通过。**这是 Linux 上运行 Windows 程序的开发证据，原生 Windows/MSVC、Windows Vulkan 和完整模型生成仍未验收。** 没有执行远程 CI，没有发布 Windows 归档，没有改动固定 ncnn 源码。

## 实现

- `tokenizer/CMakeLists.txt` 要求交叉构建明确 Rust target，使用对应的 Cargo 输出目录和目标静态库名称，导出 Windows 所需系统库。
- CPU GEMM 诊断在 Windows 使用进程峰值工作集查询，明确以 KiB 输出 `peak_working_set_kib`；测试使用 Windows PID 接口。
- 报告使用 `_wopen` 的独占创建标志，再将已拥有的描述符交给流。实际旧 Windows CRT 忽略 `fopen` 的 `x` 后缀并覆盖了旧报告，原始失败保留。回归检查既要求第二次创建失败，也检查旧内容没有被改写。
- `cli/windows_main.cpp` 接收原生 UTF-16 参数、显式转换为 UTF-8。公共 API、Rust 和组件路径字符串以 UTF-8 传递，文件边界转为原生路径，ncnn 加载使用宽字符重载。Windows 适配集中在 CLI 入口和文件边界，没有改变模型计算、权重、精度、尺寸或缓存策略。
- Unicode 检查用原生路径独立创建中文、俄文、空格和非 BMP 字符目录，再检查 CLI/API/Rust 路径、提示词、原始张量、图像和小型 ncnn 网络。它们不是完整 ERNIE 权重推理。

构建使用 MinGW GCC 16.1.1、Rust 1.98.0、`x86_64-pc-windows-gnu`、libpng 1.6.58、zlib 1.3.2、Wine 11.0。没有检测到 OpenMP，Vulkan 和紧凑读取实验均关闭。明确选择独立 Rust 工具链，未修改默认 Rust 工具链。使用方法和官方依据见 [Windows 构建说明](../../../docs/BUILDING-WINDOWS.md)。

## 实际结果与保留的失败

| 检查 | 实际结果 | 记录 |
|---|---|---|
| 省略交叉 Rust target | 配置按预期拒绝；显式设置后配置通过 | `development/build-commands.json` |
| 初次完整交叉构建 | 因 POSIX `sys/resource.h` 失败；修复后构建通过 | `development/build.log`、`build-v2.log` |
| 初次 Windows C++ 检查 | 26/27 通过，报告独占创建失败 | `development/ctest.log`、`ctest-windows.xml` |
| 独占创建修复 | 两项受影响 Windows 检查及三项 Linux 检查通过 | `development/ctest-affected.xml`、`linux-ctest.log` |
| 初次跨系统 Python 包/CLI 检查 | 59/62 通过；三项 POSIX 符号链接未被 Wine 暴露为 Windows 链接 | `development/python-windows.log` |
| 最终 Windows C++ CPU 检查 | 28/28 通过，零失败/跳过；286.83 秒 | `development/ctest-utf8-windows.xml` |
| 最终 Windows CLI/报告 | 35/35 通过，零失败/跳过；513.198 秒 | `development/python-utf8-windows.log` |
| 实际 Windows 链接 | 图片文件、图片目录、PE 文件三个场景：有效包先通过，Windows 接口创建链接后均被拒绝 | `native-links/result.json`；辅助构建 1 次、原生调用 9 次 |
| 最终 Linux CPU CTest | 35/35 通过，零失败/跳过；2.04 秒 | `development/ctest-utf8-linux.xml` |
| Linux Vulkan 构建 | 使用新代码重新构建通过；三个受影响的 CPU 执行检查通过，无 GPU 模型运行 | `development/linux-vulkan-utf8-build.json`、`ctest-utf8-vulkan-cpu.xml` |
| 迁移 SDK v1 | 安装、隔离、外部链接通过；Wine 初始化因只读临时目录失败 | `sdk-v1/result.json` |
| 迁移 SDK v2 | 临时目录修正后继续；中文/emoji 包路径触发 Windows 文件名错误 123 | `sdk-v2/result.json` |
| 迁移 SDK v3 | UTF-8 修复后 11 个命令达到预期；Wine 后台清理超时，服务整体失败 | `sdk-v3/result.json`、`sdk-v3-cleanup-journal.json` |
| 迁移 SDK v4 | 明确清理此任务独有的 Wine 前缀，全部 13 个命令达到预期，服务成功退出 | `sdk-v4/result.json`、`successful-unit-journals.json` |

v4 将带中文和空格的安装前缀移动到新位置，隐藏整个原项目与构建目录、禁用网络，使用安装后的 CMake 包构建外部消费者。故意设置的异前缀 ncnn 哨兵没有被使用；导出配置不引用旧源码、构建或安装位置。消费测试、帮助、CPU 诊断通过，损坏包、Unicode 未知参数和 Unicode 提示词文件的三个预期拒绝均返回 1，不能把这些预期非零误计为执行失败。CLI 与消费者所需的五个 DLL 有递归导入检查和实际安装散列；没有提交这些二进制，也没有作出再分发许可结论。

原始三项 POSIX/Wine 链接失败仍然存在于原记录。Windows Rust 的元数据观察显示，Linux 创建的链接在该 Wine 映射中被作为普通文件/目录暴露，而 Windows 接口创建的链接可被识别并拒绝。新增三场景没有修改或跳过原测试，也不能替代原生 Windows 文件系统验收。早期元数据辅助程序没有冻结初版源码；归档中的 `link-semantics.rs` 是用于后续三场景的扩展版，不给早期记录补造源码身份。

## 身份、资源与适用范围

最终源码以 `8158dc153d7bb4a20d2f5cf8d3d2ca764a8db524` 加 `source.patch` 表示，`source-inventory.json` 覆盖 334 个源码/构建/测试文件。早期六文件改动和 UTF-8 修复前的二进制身份另存于 `development/pre-utf8.diff`、`pre-utf8-source/` 和 `pre-utf8-binary.json`；不同阶段的日志不能全部归属于最终代码。

最终 Windows CLI 为 22,115,394 字节，SHA256 `5c1706e89d8137d408e2b2fc459e6eb34ba778b5eb27bc578e4f06e6a12e4209`。构建配置、各可执行文件/归档身份、最后导入表和固定 ncnn 身份见 `final-builds.json`、`final-executable-imports.json`、`toolchain-and-upstream.json`。ncnn 为干净的 `6a1bf000f363714839a36793addc8c879d3d899e`。初次收集的 `final-builds.json` 中 Linux Vulkan `build-dev` 仍为旧二进制；随后该目录已使用新代码重新构建，最新身份和三个受影响的 CPU 执行检查见 `final-linux-vulkan-build.json`。冻结内存对照仍使用其独立的旧 `ff2f934f...` 快照，不能归为新的 Windows 改动结果。

这些模型无关 CPU 任务限制在独立的两个 CPU 核心、4 GiB 进程组内存、零交换配额中，与原先只占用另一组核心的串行 GPU 内存对照并行。Wine 检查耗时包含启动开销，不提供 Windows CPU 性能结论。v3 服务的清理超时单独保留；v4 成功日志和测试任务终态均保留。`unit-status.json` 中已卸载的 transient unit 显示默认值，不能据此反推历史成功或内存峰值，历史终态以日志为准。

本轮没有运行完整模型、没有增加质量样本、没有测量 GPU/RAM 自动迁移性能，也没有完成 P4 或全面超过参考项目的验收。正式画质、剩余尺寸、配对速度/资源、原生 Windows/MSVC/GPU、macOS 和发布仍待相应实际证据。所有较早失败保持可见。

# 原生平台验证记录

本批次对应 README、ncnn Discussions 教程草稿和三平台原生构建验证。用户明确授权推送私有仓库的 `codex/surpass-reference` 并运行 GitHub Actions；Discussion 保持草稿。模型权重和构建产物未加入本记录。

## 执行记录

| 运行 | 源码 | 结果和目录 |
|---|---|---|
| [初始三平台](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34167327685) | `93a1b3c` | 三个 Linux 作业成功；Windows 配置失败；macOS 构建成功但 20 项 CTest 失败。`initial/` 保留配置、XML 和原始日志。 |
| [换行修复与 macOS 栈诊断](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34168231514) | `9c20901` | 三个 Linux 作业成功；Windows 已通过配置，在 `std::stoi` 缺失声明处编译失败；macOS 仍有 20 项失败。`newline-fix/` 保留日志，初始目录还保存 LLDB 完整作业日志。 |
| [macOS loader 与虚拟 GPU 配置](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34168637493) | `dbbd9f0` | 原生 CTest 53 项通过、4 项 BF16 跳过、0 失败；后续 22 项 HTTP/清单检查为 7 通过、7 失败、8 错误。见 `macos-loader-fix/`。 |
| [macOS 临时目录夹具修复](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34169350983) | `4125b25` | 作业成功，原生 CTest 53 通过、4 跳过；23 项 HTTP/清单检查全部通过。见 `macos-final/`。 |
| [Windows 直接头文件修复](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34169438372) | `237b10a` | 原生 MSVC 构建成功，CTest 33 通过、21 跳过、3 失败；23 项 HTTP/清单检查通过。见 `windows-include-fix/`。 |
| [同一源码三平台复查](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34170659886) | `3a04811` | 五个作业全部成功。Linux 两个 CPU 配置各 36/0/0；Linux Vulkan 和 macOS 各 53/4/0；Windows 36/21/0，依次为通过/跳过/失败。每个作业的 23 项 HTTP/清单检查通过。完整日志、XML、配置与运行身份见 `final/`。 |

最终源码为 `3a04811b41aa19a8d9a874981c1569816105ee26`。[summary.json](summary.json)将每个原生作业绑定到源码、运行链接、工具链、CTest 数量和 HTTP 结果。文档归档随后单独提交，不改变已经验证的源码与工作流。

单平台手动运行中的其他作业由平台选择条件跳过，与 CTest 的设备能力跳过分开记录。每个运行 JSON 包含完整的源码 SHA 和运行链接；人工表格不使用工作流总状态替代单个平台结果。

## 从 XML 独立统计

[ctest-summary.json](ctest-summary.json)逐个解析所有保留的 XML，将通过、失败、跳过及其名称分别计数。CTest 日志中的“100%”可能包含未执行的设备测试，因此不能把 57 项登记数写成 57 项实际通过。

- 本机 Linux：CPU 36 通过、Vulkan 57 通过，均无跳过或失败。`local-linux/` 保存构建、测试日志和 XML。
- Ubuntu 软件 Vulkan 与 GitHub 托管 macOS MoltenVK 作业各为 53 通过、4 项 BF16 能力跳过。跳过名称为 `residual_contract_vulkan_bf16`、`erf_gelu_vulkan_bf16`、`rmsnorm_range_vulkan_bf16`、`layernorm_range_vulkan_bf16`。
- Linux CPU 的普通读取器和可选紧凑读取器各 36 项通过。
- Windows 原生 MSVC 为 36 项通过、21 项因缺失 Vulkan 驱动跳过、0 失败；搬移安装及独立 C++ 消费者通过。跳过的完整名称保存在 XML 与统计文件中，不计作 Windows GPU 执行。
- 本机修复后的 HTTP/清单回归为 23 项通过，日志为 `local-linux/http-fixture-regression.log.gz`。

## 修复与可复查边界

Windows CRLF 只在校验前统一为 LF，完整摘要值未修改。两个换行版本的派生 shader 和读取器分别相同，三个实质源码变更均被拒绝，见 [source-newline-check.json](source-newline-check.json)。MSVC 的后续修复是为 `std::stoi` 直接包含 `<string>`。固定 PE 夹具另由 `.gitattributes` 保持 LF，实际 `core.autocrlf=true` 检出对照见 [fixture-checkout.json](fixture-checkout.json)。

Windows 的模型读取器测试原先假设 vector 元素字节数等于底层分配请求。[MSVC 大分配包含对齐空间](https://github.com/microsoft/STL/blob/main/stl/inc/xmemory)；新检查按规定元素数对同标准库空 vector 执行 `resize`，要求原生读取器请求与对照精确相等。最终 MSVC 日志记录大夹具有效元素 16,777,224 字节、分配请求 16,777,263 字节，两者相差 39 字节；原生请求与对照完全相等。FP16/BF16 所有输出位、对齐消费量和截断拒绝要求保持不变。相关三项本机 CPU 测试通过，XML 为 `local-linux/portable-fixtures-linux.xml`；最终 Windows 也全部通过。

macOS 的 LLDB 栈从 Apple Paravirtual 经 MoltenVK 图像更新、内存绑定进入 ncnn 占位图像创建，尚未执行 ERNIE 算子。使用系统 Vulkan loader 修复 SDK 源码路径泄漏和缺失 ICD 的测试行为；托管 GPU 同时设置 `MVK_CONFIG_USE_MTLHEAP=0` 后通过小型执行。这是通过的配置组合，未独立分离两个设置的因果作用，也没有修改物理 Mac 的通用运行环境。

后续 macOS HTTP 错误来自测试临时路径含系统 `/var` 软链接。测试夹具使用实际路径，新检查独立构造软链接祖先并确认拒绝；生产下载器继续保留原限制。

本机 CPU 和 Vulkan 在 CMake 修复后、直接头文件修复后分别重建成功，CLI 都与本轮最初已测二进制逐字节相同，记录在 `local-linux/*-after-build.json` 与 `*-include-build.json`。CPU SHA-256 为 `6830d305a042d5c616d2484ad3d928982e3fe73cf4f090ea04acb9f0539e2066`，Vulkan 为 `3fc8009f3083f59b5b6365f3692fde2e92afa9e06ad7f65a5b08634d28a30010`。它们与历史完整模型的冻结程序分别记录。

源码清单使用项目的 `tools/source_inventory.py` 收集全部源码、构建目录和工作流：[source-inventory.json](source-inventory.json)。固定 ncnn 提交为 `6a1bf000f363714839a36793addc8c879d3d899e`。压缩日志保留完整原文，gzip 时间戳归零便于内容复查。

## 文档验证

移植教程的 19 个代码围栏已逐项分类：Shell 语法与脚本参数检查通过，三个 Python 组装段分别构造 25 个文本模型、36 个 DiT 模型，以及 36 个 DiT 加 25 个文本模型的准备参数。检查只验证命令接口，没有重新下载或运行完整模型。记录为 [tutorial-snippets.json](tutorial-snippets.json) 与 [tutorial-python-arguments.json](tutorial-python-arguments.json)。链接、风格与排版复查见 [style-review.md](style-review.md)。本目录的文件摘要见 [file-inventory.json](file-inventory.json)。

完整模型出图、与官方的 PNG/张量差异采用各自历史实验，不由本批次 CI 补齐。Windows/macOS 完整出图、真实设备速度、激活卸载及分配失败恢复均不在本次通过范围内。

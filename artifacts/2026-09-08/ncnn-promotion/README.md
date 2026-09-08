# ncnn 最新 Git 正式升级

2026-09-08，按用户要求把正式依赖从 `6a1bf000f363714839a36793addc8c879d3d899e` 升级到 `3b7bdba7fc8aea8fd46779533eee027df77c639d`。GitHub 官方 API 在本轮确认这是上游 `master` 当前提交，提交时间为 2026-09-07 11:42:31 UTC。子模块和 `sources.lock.json` 使用同一明确 SHA；常规构建继续禁止不匹配的运行库。

## 改动范围

- 本工作树的正式 ncnn 子模块已初始化到新版，常用 `build-dev` 改用该路径，`ERNIE_ALLOW_UNPINNED_NCNN=OFF`。保留原主工作树和独立实验的旧版源码、二进制及结果。
- 模型包的来源版本与构建版本分开检查。Python 准备、验证、打包工具与原生 Rust 包验证器共用锁文件中的明确兼容名单，接受当前版本及已验证的旧 `6a1bf000`，拒绝其他版本与非法类型。重打包时保留已有来源版本，所有文件摘要、尺寸、官方模型来源、图结构和配置校验继续生效。schema-3 共享包沿用原有对象与注册表契约。
- 可选分配统计器为新版增加完整的 7,829 项源码清单摘要。`allocator.cpp` 与旧版逐字节相同，观察钩子不变；注意力 shader 和可选紧凑读取器的原始文件摘要也不变。没有关闭来源检查。
- README、运行指南、复用指南和 Discussions 草稿同步当前版本。以后升级优先选最新 Git，验证后固定明确 SHA，历史实验保留实际使用的依赖身份。

## 验证

常规 Linux/NVIDIA Vulkan 构建成功，57 项 CTest 全部通过，0 失败、0 跳过，包含原生/Python 包验证、新旧模型版本、重打包来源保留和安装后独立 C++ 消费者。模型/分配统计工具的 19 项检查、encoder 的 12 项检查、HTTP/清单的 23 项检查均通过。详见 [本机汇总](local-summary.json) 与 `local/` 的原始日志/XML。

故意传入旧运行库源码时，CMake 正确拒绝配置；兼容旧模型没有放宽构建版本锁定。第一次日志检查未识别 CMake 换行而误报，保留原始检查记录，并从同一日志重新读取确认拒绝，无需重跑。

常规程序 SHA-256 为 `288a83e8f1246dec5485554be989b97a1e36ffd8651a1bdeae4b479f21a33d7b`；安装 SDK 的 `Ernie_NCNN_REVISION` 正确报告新版。342 个源文件绑定见 [源码清单](source-inventory.json)，全部与升级提交 `a495443fa9fc031f9611e4b1f93a301656c8b423` 一致，见 [提交身份](promoted-identity.json)。后续文档归档不改变这些已验证源文件。

现存 `turbo1024-s64-portable`、`turbo-shared-v2`、`pe-cpu-v1` 三个真实模型包均由新版程序执行完整 `--verify-model`，返回 0、清单摘要保持不变。此项只读取和校验文件，未重新生成图像，见 [旧包校验记录](existing-package-checks.json)。

## 新版三平台 CI

同一源码 `a495443` 的 [Actions 运行 34242182771](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34242182771)五个作业全部成功，无需 CI 修复或重跑。运行身份见 [ci-run.json](ci-run.json)，下表由五份 CTest XML 独立统计，详见 [ci-summary.json](ci-summary.json)。

| CI 作业 | 实际通过 | 跳过 | 失败 |
|---|---:|---:|---:|
| Linux CPU，reader OFF | 36 | 0 | 0 |
| Linux CPU，reader ON | 36 | 0 | 0 |
| Linux Mesa Vulkan | 53 | 4 | 0 |
| Windows MSVC | 36 | 21 | 0 |
| macOS ARM / MoltenVK | 53 | 4 | 0 |

每个作业还通过 23 项 HTTP/清单检查；旧模型兼容和重打包来源保留均实际通过。完整作业日志明确记录检出 `3b7bdba7`。`ci/` 保存每个作业的配置、XML、CTest 原始记录和完整作业日志，文本日志以 gzip 保留原始字节。

Linux Mesa 和托管 macOS 仍报告 `bf16-p/s=1/0`，四项 BF16 测试因原生 storage 能力缺失而跳过。Windows 的 21 项跳过来自托管机器缺少 Vulkan 驱动。跳过不计作通过，也不属于显存或 RAM 不够；本机 NVIDIA 的对应 57 项均实际通过。CI 不下载完整 ERNIE 模型，不代表 Windows/macOS 完整出图或性能已验证。

## 已有完整出图依据

此次采用的就是[上一轮独立 A/B 实验](../ncnn-latest-git/README.md)中的候选提交。57 项本机 CTest 全过；9 对真实权重输出逐位相同；512×512、8 步英文苹果样例在 FP32/FP16/BF16 下分别有 25 个张量文件及 PNG 与旧版逐字节相同。五次完整新执行均正常退出、没有 OOM。此次只调整依赖和版本兼容元数据，没有重跑这些已完成的模型实验。

原官方诊断结果仍为 FP32 25/25、FP16 23/25、BF16 18/25；PNG 最大通道差仍为 1、109、98。新版独立 BF16 Reduction 抵消测试有所改善，但该图片没有变得更接近官方。这里没有新增速度、广泛画质或 Windows/macOS 完整模型推理的结论。

准备过程曾尝试将浅克隆作为 Git reference 被拒绝；随后独立初始化子模块成功。一次候选浅克隆缺少旧提交，导致源码差异查询失败；改在包含两个提交的正式子模块中确认 `allocator.cpp` 无差异。两者均发生在构建与模型执行之前，不属于运行库回归。

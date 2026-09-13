# 三平台程序包与 Hugging Face 下载入口

公开 HF 版本为 `1f614857c117ea9a1cf814e639bd514154973c18`。本轮把预编译程序加入现有 [Hugging Face 模型仓库](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn)。用户选择平台 ZIP、完整解压，再运行 `python3 run.py --prompt ...`；Windows 使用 `python`。首次运行下载并校验主模型，之后使用本地文件。

运行时代码与打包脚本固定为 `d2fd973f7658de5162e8edfee13171bb2e6c2e15`，ncnn 仍为 `3b7bdba7fc8aea8fd46779533eee027df77c639d`。本轮没有更改 C++ 推理实现、模型转换或原权重。主包与 PE 的固定下载清单仍指向原 HF 版本 `9924de97ebe85c540ce09e207142a6efee614be7`。

## 下载内容

| 程序包 | 字节数 | ZIP 内文件数 | 平台 |
|---|---:|---:|---|
| `ernie-image-linux-x86_64.zip` | 12,549,739 | 234 | Ubuntu 24.04 / glibc 2.39、GCC 13 C++ 运行库兼容系统 |
| `ernie-image-windows-x86_64.zip` | 7,409,475 | 205 | Windows 10/11 x64 |
| `ernie-image-macos-arm64.zip` | 7,595,667 | 205 | macOS 15 Apple Silicon |

ZIP 包含原生命令行程序、所需的非系统运行库、首次下载与启动脚本、固定模型清单、许可证、源码和 CI 身份、文件校验清单。模型权重单独存放：主包 23,271,740,211 字节，PE 7,680,869,430 字节；只在加 `--with-pe` 时自动下载 PE。

Linux 包附带 PNG、zlib、OpenMP，保留系统 glibc/C++ ABI 和 Vulkan 驱动依赖。Windows 包附带 MSVC 运行库。macOS 包附带 PNG、Vulkan loader 和 MoltenVK，动态库及 ICD 使用包内相对路径；程序采用 ad-hoc 签名。Python 3.10+ 只用于可选下载和启动脚本，原生推理不依赖 Python 包。

同一主包可选择 `--precision fp32|fp16|bf16`；BF16 仍为实验选项。宽高为 16..2048 范围内的 16 的倍数，总面积最多 2,097,152 像素。`turbo/objects/` 和 `pe/block-*/` 是正常的模型组织方式，程序自动加载，不需要用户重命名或合并。

## 验证

[最终 CI 34772511807](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34772511807) 五个原生任务均成功。框架 CTest 合计 **244 项通过、40 项按设备能力跳过、0 失败**；三个平台上还分别检查了程序包搬到独立目录后的 `run.py --help` 和 `--diagnose`。

| 配置 | CTest 通过 | 设备能力跳过 |
|---|---:|---:|
| Linux CPU、普通 reader | 40 | 0 |
| Linux CPU、紧凑 reader | 40 | 0 |
| Linux Mesa Vulkan | 62 | 6 |
| Windows MSVC | 40 | 28 |
| macOS MoltenVK | 62 | 6 |

Linux/MoltenVK 的六项跳过对应 CI 后端未提供的 BF16 能力；Windows 托管 runner 未提供可用 Vulkan 设备，28 项设备测试跳过。这些是 CI 设备环境的覆盖边界，原生 CPU、CLI、包契约和构建均通过。JUnit 记录随本目录及每个平台 ZIP 保存。

五个任务各执行 38 项下载清单、HTTP 下载和启动脚本检查，合计 190 次通过。启动脚本覆盖首次准备、复用完整包、不自动下载 PE、自定义路径、中文参数、错误退出码、无效尺寸提前拒绝和仅校验 PE 等行为。

下载三个 CI ZIP 后，独立核对归档 SHA-256 和包内全部 644 个文件。Linux ZIP 另外实际解压到带空格的独立路径，用其中的程序和已发布主包完成了一张完整图片。

## Linux 下载包的完整出图

![本次 Linux 程序包生成的 512×512 红苹果](linux-image/apple.png)

提示词：`A red apple on a wooden table, soft daylight, realistic photo.`

- 使用归档内 `run.py`，只传模型目录、提示词、输出路径、报告路径和 `--threads 2`。图像 512×512、Vulkan FP16、8 步、seed 42 均来自默认设置，PE 关闭。
- 测试机器为 RTX 4060 Laptop 8GB。程序记录的生成与写图耗时 **241.1888 秒**；外层受限执行范围 **241.8891 秒**，计时范围不同。
- cgroup 内存峰值 **13,019,090,944 字节**，包括文件缓存，不能视为进程 RSS。整块显卡的采样峰值 **2103 MiB**，包括桌面及其他进程。
- 既有边界为 16 GiB cgroup、禁用该范围 swap、2 个 CPU 核配额、3 GiB 系统可用内存下限和 6144 MiB 整卡显存保护。`max`、`oom`、`oom_kill` 事件均为 0，运行时恢复重试为 0。
- 实际选择 15 token、32 文本桶、64 DiT 文本槽，完成文本编码、全部 8 个去噪步骤、CPU VAE 和图片写入，返回码为 0。
- PNG SHA-256：`45554964e13151c9daa9bdb61c627f56417839d00031417b8aa1da1d1e43ecba`。

这次执行用于验证发布程序的完整使用流程，没有重跑官方数值对照，也不作为不同配置或项目之间的性能比较。既有精度误差继续见 [NUMERICAL-RESULTS.md](../../../docs/NUMERICAL-RESULTS.md)。

## 首次打包问题与修正

[首次 CI 34771913462](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34771913462) 的原生测试通过，程序打包阶段暴露了两项问题：Linux 的 glibc `libmvec.so.1` 未列入系统库；完整 Cargo.lock 的许可证清单把只面向 UEFI 的 `r-efi` 当成了桌面程序依赖。

修正后把 `libmvec` 归入系统 glibc，并根据锁定的当前 Rust target 依赖图判断必需的许可证。三个目标图都不包含 `r-efi`，原来的缺项仍记录在归档的排除清单中。没有关闭检查或改动模型计算。首次失败日志压缩保留在本目录，最终同源 CI 五项全部成功。

HF 公开版本、匿名下载核验和现有 Discussion 更新的精确回执见本目录的 `publication.json`、`remote-inventory.json` 和 `discussion-receipt.json`。上传只增加程序包并更新入口和元数据，原有 150 个模型文件保持不变。

现有 [ncnn Discussion #6985](https://github.com/Tencent/ncnn/discussions/6985) 已于 2026-09-13T19:33:07Z 更新下载链接及运行命令。正文逐字读回和匿名页面检查通过，原标题、分类及三张原图链接均保留。

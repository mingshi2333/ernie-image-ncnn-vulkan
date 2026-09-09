# 平台验证

本页分别记录原生构建、小型执行测试和完整模型出图。CTest 的“通过”数排除设备能力导致的跳过项；未下载 ERNIE 权重的 CI 只覆盖构建、接口和小型网络。

## 2026-09-09 内存执行最终验证

后台预取、DiT RAM buffer 与检查点恢复的最终代码为 `7296bfeba2c809d54a7d6214b0f08dee861ad1f8`，ncnn 仍为 `3b7bdba7`。[CI 34373124004](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34373124004)五个作业全部成功；下载后的原始 XML、日志与产物摘要独立核对了实际检出源码和下面的数量，见[完整归档与机器可读统计](../artifacts/2026-09-08/memory-execution/v3/final-ci/summary.json)。

| 环境 | CTest 实际通过 | 跳过 | 失败 | 验证范围 |
|---|---:|---:|---:|---|
| 本机 Linux / RTX 4060 Laptop | 62 | 0 | 0 | 真实 Vulkan 小型网络、预取、RAM buffer、故障恢复及 BF16；Khronos validation 无错误 |
| 本机 Linux 纯 CPU | 37 | 0 | 0 | 内存执行实现的独立 CPU 构建；最终源码 CPU 另由下方两个 CI 作业覆盖 |
| Ubuntu 24.04 CPU，读取器 OFF / ON | 各 37 | 0 | 0 | 两个最终源码的独立 CPU 构建 |
| Ubuntu 24.04 / Mesa 软件 Vulkan | 57 | 5 | 0 | 小型 Vulkan 实际执行；5 项缺少 BF16 storage，运行前跳过 |
| Windows Server 2025 / MSVC x64 | 37 | 25 | 0 | 原生构建、CPU、CLI、安装 SDK；25 项因没有 Vulkan 驱动跳过 |
| macOS 15 ARM / Apple Clang / MoltenVK | 57 | 5 | 0 | 托管虚拟 GPU 的小型 Vulkan 执行；5 项缺少 BF16 storage |

远程合计 260 项 CTest：225 项通过、35 项跳过、0 项失败；五个作业另有各 23 项 HTTP/清单测试通过。新增 `bf16_gemm_vulkan` 在具备原生 BF16 storage 的本机实际执行，在 Mesa/MoltenVK CI 上按能力返回 77。这些跳过是 **CI 驱动能力限制，与机器 RAM 大小无关**，不能计为对应算子已执行通过。

此次最初的 Linux Vulkan CI 失败另有原因：Ubuntu 校验层 1.3.275 不识别新版 ncnn 的 `VK_KHR_shader_subgroup_rotate` 特性结构，而 Mesa 设备已支持它。保留[首轮失败](../artifacts/2026-09-08/memory-execution/v2/first-ci/summary.json)，改用按官方 SHA256 固定的 Linux SDK 1.4.357.1，仅提取隔离的校验层与 `vulkaninfo`，不替换系统 loader/Mesa。最终日志确认实际加载 layer 1.4.357，设备仍为 Mesa 25.2.8 / llvmpipe LLVM 20.1.2 / Vulkan 1.4.318，系统 loader API 仍为 1.3.275。预检与独立 VUID 扫描全部保留；前一修正版本 `5b57e9d` 的[五项成功记录](../artifacts/2026-09-08/memory-execution/v2/second-ci/summary.json)也保留。

最终 CI 中只有 Linux Vulkan 作业启用 Khronos validation。所有保留日志均无 VUID，但不能据此声称其他平台做过同等校验层验证。macOS 内存小型测试可注入已知主机余量，生产环境尚无 macOS RAM 余量读取器；Windows Job/Wine 同样拒绝无法确认余量的 host 准入。

完整模型另在 Linux 上串行执行：正常与混合内存 FP32 各 25/25 张量及 PNG 与旧基线逐位一致；默认 FP16 同样与旧 FP16 逐位一致，官方仍为 23/25、PNG max 109；修正后的 BF16 零 VUID，官方仍为 17/25、PNG max 143。详见[完整实验及独立回读](../artifacts/2026-09-08/memory-execution/README.md)。**Windows/macOS 完整 ERNIE 出图、物理 Mac、真实显存耗尽和跨平台性能仍未在本次验证。**

## 2026-09-08 验证结果

[三平台工作流](../.github/workflows/build.yml)构建 CLI、原生 tokenizer 和可安装的 C++ SDK。它在 `main`、`codex/surpass-reference` 推送、Pull Request 或手动运行时触发；手动运行可选择单个平台。下表的该次五个 CI 作业均来自升级源码 `a495443`、ncnn `3b7bdba7`，原始平台失败与旧版成功记录保留在[先前日志归档](../artifacts/2026-09-08/native-platforms/README.md)，新版证据保存在[升级记录](../artifacts/2026-09-08/ncnn-promotion/README.md)。

| 环境 | CTest 实际通过 | 跳过 | 失败 | 范围 |
|---|---:|---:|---:|---|
| 本机 Linux CPU（先前 `6a1bf000`） | 36 | 0 | 0 | 保留旧版独立 CPU 记录；新版 CPU 构建见下方两个 CI 作业 |
| 本机 Linux / NVIDIA Vulkan | 57 | 0 | 0 | 含实际 Vulkan 算子、缓存及 BF16 小型测试 |
| Ubuntu 24.04 CPU，读取器 OFF / ON | 各 36 | 0 | 0 | 两个独立构建均通过 |
| Ubuntu 24.04 / Mesa 软件 Vulkan | 53 | 4 | 0 | CI 软件驱动缺少原生 BF16 storage，4 项跳过；其余 Vulkan 测试实际执行 |
| Windows Server 2025 / MSVC x64 | 36 | 21 | 0 | 原生编译、CPU、CLI、SDK 通过；运行器无 Vulkan 驱动，21 项设备测试跳过 |
| macOS 15 ARM / Apple Clang / MoltenVK | 53 | 4 | 0 | GitHub 托管 Apple Paravirtual GPU 上的小型 Vulkan 执行；CI 驱动缺少原生 BF16 storage，4 项跳过 |

源码 `a495443fa9fc031f9611e4b1f93a301656c8b423` 的[三平台原生构建与小型测试](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34242182771)五个作业全部成功；每个作业另有 23/23 项本地 HTTP 与清单测试通过，并执行了旧模型包兼容与重打包来源保留检查。数量从下载后的 CTest XML 独立统计，跳过项排除在通过数之外，详见[机器可读结果](../artifacts/2026-09-08/ncnn-promotion/ci-summary.json)。每个作业的日志均确认实际检出新版 ncnn，具体工具链保留在配置与完整日志中。此前 Windows 的三项测试失败及其修复记录仍保留，未用新版结果覆盖。

**这 4 项 BF16 跳过由 CI 环境的驱动能力限制导致。** Linux 的 Mesa 软件设备和 macOS 的托管虚拟设备均报告 `bf16-p/s=1/0`：ncnn 的 BF16 打包路径可用，原生 BF16 storage 不可用。残差加法、ERF GELU、RMSNorm、LayerNorm 的对应测试在运行前检查能力，返回 `77` 后由 CTest 标记为跳过。本机 RTX 4060 Laptop 报告 `bf16-p/s=1/1`，同样四项已实际执行并通过。

**完整模型出图目前有 Linux 实测记录，Windows 和 macOS 尚未运行完整 ERNIE 模型。** [七种尺寸与提示词误差](../README.md#与官方的实测误差)采用各自已保存的真实权重实验，保持原始数值门槛和未通过项。本轮 CI 没有重新运行这些大模型实验，也不提供三平台速度排名。较早的 [MinGW/Wine 验证](../artifacts/2026-09-07/windows-cpu-portability/README.md)属于单独的交叉构建与兼容层证据。

## 此次原生验证修复了什么

- **Windows 的源文件换行。** 固定 ncnn 源码检出为 CRLF 后，原始文件摘要与 LF 基线不同。派生前统一换行，再用原来的完整 SHA-256 校验同一文本。两个换行版本生成相同结果，三个真实源码改动仍被拒绝，见[五项检查](../artifacts/2026-09-08/native-platforms/source-newline-check.json)。
- **Windows 的固定 PE 夹具。** 通过 `.gitattributes` 保证有固定字节摘要的 `.jinja` 与 `.param` 夹具检出为 LF；实际 `core.autocrlf=true` 检出后，两份文件仍与原始字节相同，见[检出对照](../artifacts/2026-09-08/native-platforms/fixture-checkout.json)。
- **MSVC 的直接头文件依赖。** `model_config.cpp` 补充 `<string>`，使 `std::stoi` 不再依赖其他标准库头文件的间接包含。
- **MSVC 的 vector 分配观测。** [MSVC 的大 vector 分配](https://github.com/microsoft/STL/blob/main/stl/inc/xmemory)包含额外对齐空间，不能直接与元素有效载荷字节数相等。测试增加同一标准库的指定元素数对照，仍要求原生读取器的实际分配精确匹配，并保留全部逐位解码和截断拒绝检查。
- **macOS 的 Vulkan 加载与 SDK 搬移。** 默认关闭 ncnn 的 `NCNN_SIMPLEVK`，通过系统 Vulkan loader 链接并指定 MoltenVK ICD，避免安装后的 SDK 引用原源码目录内的 `.tbd`；缺失驱动测试也由 loader 按 ICD 配置处理。
- **托管 Mac 的虚拟 GPU 分配。** LLDB 定位到 ncnn 创建占位图像时的 MoltenVK/Apple Paravirtual 崩溃。CI 同时采用上述 loader 和 `MVK_CONFIG_USE_MTLHEAP=0` 的普通 Metal 分配配置后，小型 Vulkan 测试通过。这一环境设置没有加入通用运行入口；物理 Mac 和完整模型行为仍需实机验证。
- **macOS 的临时目录夹具。** `/var` 指向 `/private/var`，下载测试改用实际临时目录，并新增故意构造软链接祖先的拒绝检查。生产下载器的软链接限制保持不变。

没有修改 ERNIE 数学公式、推理默认精度或模型张量与图像的数值检查阈值。macOS loader 默认值属于平台构建修复；托管虚拟 GPU 的设置只用于 CI。最初的构建失败、崩溃、安装失败和下载测试失败均可在归档中复查。

## 覆盖与复现

公共 CTest 覆盖核心算子、原生缓存、API/参数、Unicode 路径、图片 I/O、模型包拒绝、搬移安装和外部 C++ 消费者。返回码 77 表示 GPU 驱动或精度能力不可用；这类测试保留名称和跳过原因。另有 Python 本地 HTTP 服务测试下载、校验和发布清单，不需要下载模型。

[Windows 原生 MSVC 命令](BUILDING-WINDOWS.md#原生-msvc-构建)与工作流使用同一 Rust target 和 libpng/CRT 组合。macOS 可在初始化子模块后执行：

```sh
brew install cmake ninja libpng vulkan-loader molten-vk
rustup toolchain install 1.98.0 --profile minimal
export RUSTUP_TOOLCHAIN=1.98.0
export VK_DRIVER_FILES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
cmake -S . -B build/macos -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DERNIE_ENABLE_VULKAN=ON -DNCNN_SIMPLEVK=OFF -DNCNN_SYSTEM_GLSLANG=OFF \
  -DERNIE_BUILD_TOKENIZER=ON -DERNIE_BUILD_GENERATOR=ON -DERNIE_INSTALL_SDK=ON \
  -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF
cmake --build build/macos --parallel 2
ctest --test-dir build/macos --output-on-failure
```

GitHub 的 Apple Paravirtual 运行器还设置 `MVK_CONFIG_USE_MTLHEAP=0`；上面的物理 Mac 命令没有强制这个设置。该参数定义见 [MoltenVK 配置说明](https://github.com/KhronosGroup/MoltenVK/blob/v1.4.2/Docs/MoltenVK_Configuration_Parameters.md#mvk_config_use_mtlheap)。本次没有进行 loader 和 Metal heap 设置的独立因果对照，只记录经过验证的组合。

运行器信息来自 [GitHub 运行器说明](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)、[Windows 镜像清单](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md)和 [macOS 镜像清单](https://github.com/actions/runner-images/blob/main/images/macos/macos-15-arm64-Readme.md)；具体工具链以归档日志为准。首次模型准备和生成命令见[运行说明](RUNNING.md)。

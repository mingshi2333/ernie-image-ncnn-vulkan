# P4 平台与交付 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让新用户可以下载、检查、安装并离线生成；平台支持来自实际运行记录。

**Architecture:** 构建、安装、模型准备与推理分离。可分发包由固定来源构建，CLI 无网络依赖；模型下载是独立显式工具。测试小模型只验证程序与算子连接，真实模型验收单列。

**Tech Stack:** CMake/CTest/CPack、固定 Rust 工具链与依赖、Linux GCC/Clang、Windows MSVC、macOS AppleClang/MoltenVK、GitHub Actions。

## Global Constraints

继承 [总计划](2026-09-06-surpass-reference.md)。以下命令/文件为待实现任务。平台依赖版本在实施时核对官方来源并固定；不能把当前 Linux 工作流文件存在写成远程通过。公开推送/上传/发布在候选材料完成后按用户授权执行，本阶段先完成所有可审查的本地工作。

## Task D1: 构建矩阵与安装后的使用

**Files:** Extend `.github/workflows/build.yml`, `CMakeLists.txt`, `cmake/Dependencies.cmake`, `src/CMakeLists.txt`, `cli/CMakeLists.txt`, `tokenizer/CMakeLists.txt`, `tests/CMakeLists.txt`；Create `CMakePresets.json`, `cmake/ErnieConfig.cmake.in`, `cmake/Packaging.cmake`, `tests/install/CMakeLists.txt`, `tests/install/main.cpp`, `tests/test_install.py`。

**Interfaces:** 安装导出 `ernie::pipeline` 与对应 package config；外部使用 `find_package(Ernie CONFIG REQUIRED)`。构建 preset 为 `linux-cpu`, `linux-vulkan`, `windows-vulkan`, `macos-vulkan`；二进制目录、依赖查找和 runtime 库拷贝由平台配置处理，不能在公共 C++ API 中出现平台宏。

- [ ] 先写安装消费者：从新的外部 build 目录只使用 install prefix，编译一个标准 C++ 调用程序；移走/隔离 source 和原 build 搜索路径后仍能链接运行。缺 DLL/动态库应直接使测试失败。
- [ ] 增加 Linux CPU/Vulkan、Windows MSVC CPU/Vulkan、macOS arm64 Vulkan 构建矩阵；固定所用 action commit、编译器/SDK/Rust 版本和包管理 lock。Linux 可用 Clang 做附加编译验证，不扩张为无边界编译器矩阵。
- [ ] CI 执行 CTest 与 Python 契约测试，安装后 `--help`、`--diagnose`、损坏包拒绝，以及外部 C++ 消费者。模型相关测试用自生成小网络/合成数据，避免 CI 下载全部真实权重。
- [ ] 确认 CPU-only 不链接 Vulkan，公共库不链接 PNG/JPEG；Windows 使用正确 CRT/Rust target 与 UTF-8 路径，安装 prefix 含空格/中文也能调用；macOS 显式打包或声明 MoltenVK 运行依赖。
- [ ] 对每个平台运行小网络的真实算子/完整数据连接；没有兼容设备而跳过时保留 skipped，不能计为 GPU 通过。Linux 实际 NVIDIA、Windows 实际 GPU 全模型运行在 D3 执行。
- [ ] Linux 本地执行 `cmake --preset linux-vulkan`、对应 build/CTest、`cmake --install build/linux-vulkan --prefix outputs/install-candidate`；再执行 `cmake -S tests/install -B outputs/install-consumer -DCMAKE_PREFIX_PATH="$PWD/outputs/install-candidate"` 与消费者测试。新 preset 不删除原 `build/ernie-image` 入口兼容。
- [ ] 提交 `build: validate cross-platform installation and public C++ consumption`；本地和远程状态分别报告。

**Acceptance:** 安装后的程序和库独立于源码工作区；所有必需 CI job 在实际授权触发后通过。仅写出 workflow 不关闭 D1 远程验证项。

## Task D2: 模型获取与可分发候选

**Files:** Create `tools/download_model.py`, `tools/build_release.py`, `tools/release_manifest.py`, `tests/test_model_download.py`, `tests/test_release_manifest.py`, `docs/RELEASING.md`；Extend `tools/package_model.py`, `tools/pe_package.py`, `sources.lock.json`, `README.md`, `docs/RUNNING.md`。

**Interfaces:** `download_model.py --manifest FILE --output DIR` 独立下载 schema-3 图像包或 PE 包；`build_release.py --install PREFIX --output NEWDIR` 创建平台压缩包、来源/依赖清单、SHA256SUMS 和使用说明。推理二进制继续不发起下载，也不要求用户安装 Python；下载工具仅是模型准备选项，另提供浏览器下载的分片与清单。

- [ ] 用本地 HTTP fixture 测 Range 恢复、服务器忽略 Range 返回 200、对象版本变化、截断/错误末字节、磁盘空间不足。只将完整 hash 通过的 `.part` 原子改为最终文件，禁止直接把断点文件当可用权重。
- [ ] manifest 包含固定模型 revision、文件大小/散列、图 schema、所需能力、转换来源、转换脚本 hash 和许可；下载总大小及预计落盘位置在开始前显示。恢复既有已校验文件不重复下载。
- [ ] 从所有权重重建对应规范化散列，并跑原生 `--verify-model` / PE 校验；验证 tail-layer 损坏和缺 encoder 同样被拒绝。迁移旧包不丢失 provenance。
- [ ] 建立完整 runtime 依赖/许可证清单；只打包有明确可再分发依据的代码、tokenizer 与权重资产。分开项目 MIT、ncnn、Rust/图像库、模型与提示示例来源，记录需保留的 notices。
- [ ] release candidate 只包含程序、必要库、许可和实际可执行说明；模型单独分片，使用一套共享权重，避免将 20+ GiB 模型塞入源码 Git。保留网络断开时的人工获取/校验流程。
- [ ] 在输出目录生成可审查的 release 草稿、版本/资产清单与平台支持表。外部动作尚未授权时保留本地候选，发布状态为 pending，不能编造 release URL。
- [ ] 执行对应 unittest 与迁移/完整性回归；提交 `build: assemble verifiable offline runtime and model releases`。

**Acceptance:** 下载恢复/完整性行为可靠，离线推理安装包可移动，候选资料与许可齐全；公开地址的实际可用性由 D3/P5 核实。

## Task D3: 在干净环境完成真实使用链

**Files:** Create `tools/check_release.py`, `tests/fixtures/release-smoke.json`；Extend `docs/RUNNING.md`, `docs/RELEASING.md`, `docs/CODE-STRUCTURE.md`；证据写入本次 `artifacts/<执行日期>/delivery-candidate-v1/`。

- [ ] Linux 与 Windows 各使用独立安装目录，执行包检查、短 prompt 1024 文生图、长中文非正方形生成、PE 文生图、strength=.5 图生图；必须在实际 8GB 级 GPU 上完成，保存机器/驱动/包/输入与输出 hash。
- [ ] 环境中不使用开发 `.venv`，推理阶段断网；模型目录移动、路径含中文空格、输出目录只读、已存在文件、无 GPU 和缺模型均有明确结果。
- [ ] macOS 在 arm64/MoltenVK 上完成安装消费者、小模型执行与真实图像文件读写；若没有全模型运行证据，平台表写“构建与小模型验证”，不写完整 ERNIE 已支持。
- [ ] 重复三次 PE/图生图任务，确认每次结束释放缓存/权重，峰值不随次数增长；不做未授权长时间压力测试。
- [ ] 全部本地资料完整后，按用户授权进行外部发布；之后从发布地址重新下载到新的目录，比对公开 checksum 并至少在 Linux 完成离线生成。Windows 全模型资料不能由 Wine 或 Linux 交叉编译代替。
- [ ] 如果缺 Windows/macOS 实机、人工评审或发布授权，记录具体未完成项，继续完成其他独立本地交付工作；缺少的证据仍阻止对应验收，不把环境缺失当代码通过。
- [ ] 提交 `docs: record verified platform installation and release evidence`，只记录实际结果。

**Acceptance:** 用户从公开候选实际走通相应平台的获取、校验、安装、生成；外部条件尚缺时 D 保持 incomplete。

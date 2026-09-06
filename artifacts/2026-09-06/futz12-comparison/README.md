# futz12 ERNIE 直接比较证据

日期：2026-09-06。完整结论和固定源码链接见 [直接比较](../../../docs/FUTZ12-COMPARISON.md)。

本项目运行时基线 `5ba49323549582238bafba7869b8e2e1a2f5c042`；对方源码 `8dcd6e4411137d8abe92c9d78581c4c96d5182c6`，模型元数据 `140a052f7919f279de7f697fa54f33bd1c0cac2b`。本轮没有修改运行时，也没有运行对方完整图像生成器。

- 对方 18 个源码/提示词文件按完整 Git blob SHA1 核对，并记录 SHA256。只读取 7 个模型图/分词元数据文件，没有下载神经网络二进制权重。
- 使用完全相同的 131072 项基础词表，独立编译对方原始 BPE 实现，复现 TextEncoder 的调用选项。
- 本项目原生分词对官方 53/53；对方 51/53。差异限于混合空白与特殊 token 字面量两个样本；对方自己公开的五个提示词，双方都为 5/5。
- 五个示例的 token 数为 1080、584、648、517、95（含 BOS），全部超过本项目当前 64-token 生成包容量。这是输入容量结论，不是完整生图运行结果。
- 在完整 Git tree 所证明的根目录布局上复现 README 的 CMake 配置入口，返回 1，原因为根目录缺少 CMakeLists.txt。真正入口为 src/CMakeLists.txt；尚未继续验证对方完整构建。

[local-checks.json](local-checks.json) 保存 53 项结果、词表一致性、图层统计、两个原生分词器的二进制散列、命令和日志散列。[source-manifest.json](source-manifest.json) 与 [model-metadata-manifest.json](model-metadata-manifest.json) 保存固定下载地址与文件身份。[upstream-state.json](upstream-state.json) 保存本轮查询的 HEAD、模型版本和空 Releases 列表。[tokenizer-build.json](tokenizer-build.json) 记录隔离编译命令和编译器版本。

核对脚本和最小驱动保存在 harness/；本轮使用的工作目录为 outputs/futz12-comparison-v1，准确源码、词表和完整 token 输出仍留在该目录。原有 48 个输入来自 outputs/tokenizer-v2，并逐个核对输入散列。harness 是本轮工作目录的复现记录，直接在本小型归档目录执行不会自动下载或构造所需输入。第三方完整源码、词表、提示词正文和可执行文件没有并入此归档。

本次结果不能证明双方速度、峰值内存或生成质量的胜负；这些仍需同机、同权重、同初始 latent、同精度/尺寸/步数的完整运行。

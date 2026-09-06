# 与 futz12/ernie-image-ncnn-vulkan 的直接比较

核对日期：2026-09-06。对方当前 HEAD 为 `8dcd6e4411137d8abe92c9d78581c4c96d5182c6`，提交日期为 2026-07-16；本项目比较基线为 `5ba4932`。本轮检查完整 Git tree、固定版本源码、公开模型图和分词资产，并编译运行对方原始分词器。没有运行对方完整生成器、下载其神经网络二进制权重或测量图像速度。

**对方的实用功能覆盖明显更完整。本项目目前的额外价值主要是可追溯转换、自动数值复核、已验证的分块内存策略及部分算子/分词准确性。尚无相同条件的完整生成对比，不能宣称图像质量或速度领先。**

## 功能和交付

| 项目 | futz12 当前公开实现 | 本项目当前实现 | 判断 |
|---|---|---|---|
| 基础文生图 | README 有 8 步 Turbo BF16 示例，代码含完整生成流程 | Linux Turbo、8 步、CFG=1、1024 PNG 已实测 | 两者均有基础流程；本轮未复跑对方示例 |
| 长提示词 | 原生 BPE，TextEncoder 动态输入，最多 2048 tokens，超长截断；支持 UTF-8 prompt 文件 | 官方 Tokenizers Rust 静态桥接；生成包仅支持 32/64-token 桶，超过桶容量拒绝 | 对方的提示词容量和文件入口领先 |
| 图像尺寸 | 接受正数且能被 16 整除的宽高；公开示例包含非正方形图片 | 模型包固定尺寸；已有 64×64 验证包和 1024×1024 生成包 | 对方更灵活；不能把输入检查解释为任意尺寸都已验收 |
| PE 提示词增强 | 已实现 CPU FP32、26 层、自回归采样、temperature/top-p，含 PE 示例图 | 尚未接入完整真实 PE | 对方已具备我们缺少的功能 |
| 图生图 | CLI `--input-image` / `--strength` 接入 VAE encoder、噪声混合和起始去噪步 | 尚未实现 | 源码确认对方已接通；本轮未做成图验证 |
| 精度 | 默认 BF16 storage/packing，FP16 关闭；`--fp32-storage` 可关闭 BF16 | DiT 默认 FP16，保留 FP32 残差；可选 FP32；BF16 尚未接入完整 CLI | 对方具备 BF16 全流程；我们的 FP16 需要额外溢出保护 |
| 文本和 VAE 设备 | 默认随 Vulkan 选项运行，PE 单独强制 CPU | 当前验收路径为 CPU 文本、Vulkan DiT、CPU VAE；Vulkan VAE 实验入口未完成验收 | 对方的默认 GPU 覆盖更广，不据此宣称速度优势 |
| 输出与调试 | PNG/JPG/BMP/TGA；输入 latent、条件特征、逐步 dump、VAE 单独解码、起止步和自检 | PNG、输入 latent/特征、逐步与逐层 trace、独立误差复核 | 对方并非只有演示图，也有实际调试能力 |
| 预转换模型 | Hugging Face 已提供 DiT、文本、VAE encoder/decoder、PE | 本地已完成约 21.67 GiB 的独立包；提供转换/打包工具，尚未发布下载包 | 对方模型获取方便；本项目转换来源和文件契约更明确 |
| 构建与平台 | README 声明 Windows/Linux/macOS，源码有 Windows 路径处理；当前文档构建入口有误；无 GitHub Release | Linux 硬件/软件 Vulkan、CPU 本地构建通过；安装和文件校验已完成；远程 CI 与其他平台待验证 | 双方都有交付缺口，不能只凭平台徽章认定跨平台验收 |

来源：[README](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/README.md)、[CLI 全部入口](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/main.cpp#L56)、[配置与 TextEncoder 上限](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.h#L73)、[生成器和图生图](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.cpp#L1655)、[固定版本模型目录](https://huggingface.co/wuyex/ernie-image-ncnn/tree/140a052f7919f279de7f697fa54f33bd1c0cac2b)。本项目状态见 [README](../README.md) 和 [完整轨迹报告](../artifacts/2026-09-06/attention-parity/README.md)。

本轮查看的生成函数只调用 conditional prediction，没有 negative/unconditional 分支或可配置 CFG。CLI 默认 50 步不能证明已经完整支持官方非 Turbo/CFG 模型。对方也不宜描述为覆盖全部官方功能。

## 五条公开提示词揭示的实际差距

使用相同的官方分词配置，并用本项目原生分词程序逐项复核 token IDs：

| 对方示例 | tokens（含 BOS） | 我们当前 64-token 生成包 |
|---|---:|---|
| 表情包合集，`prompt.txt` | 1080 | 超出容量 |
| 微缩城市，`prompt2.txt` | 584 | 超出容量 |
| Minecraft 信息图，`prompt3.txt` | 648 | 超出容量 |
| 新闻画面，`prompt4.txt` | 517 | 超出容量 |
| PE 前的赛博朋克描述，`prompt5.txt` | 95 | 超出容量 |

**对方展示的五条原始提示词，我们当前的生成包一条也不能直接接受。** 这里的限制来自文本模型桶；本项目分词器本身能正确处理这些输入。更长文本不是只加一个 CLI 参数，必须准备相应文本/DiT 图并验证内存与完整生成。

## 本轮实际执行的分词比较

编译对方未修改的 `bpe_tokenizer.cpp`，以 TextEncoder 中完全相同的加载参数、BOS、byte-level regex、`ignore_merges` 和 2048-token 截断调用。对方公开词表与固定官方词表的 **131072 个基础条目全部相同**，因此下面的差异不是用错词表造成的。

| 输入集合 | 本项目原生 IDs 对官方完全一致 | futz12 原生 IDs 对官方完全一致 |
|---|---:|---:|
| 已有 48 个分词回归样本 | 48/48 | 46/48 |
| 对方五条公开提示词 | 5/5 | 5/5 |
| 合计 | 53/53 | 51/53 |

两项差异均为边界样本：

- 混合空白 ` \t\n\r\n `：官方把开头的空格与 tab 编为同一个 token；对方拆成两个。token 总数为官方 6、对方 7。
- 特殊 token 字面量 `<s>hello</s><pad><unk>`：官方识别这些特殊标记；对方 TextEncoder 配置把其中标记当普通文本分词。token 总数为官方 6、对方 12。

这是隔离分词模块的实际结果，不是完整生成器的准确率或成图质量排名。对方自己的五条示例在本轮全部分词一致。本项目当前的 64-token 容量限制仍然存在，不能用分词回归通过抵消功能缺口。[原始分词器](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/bpe_tokenizer.cpp)、[TextEncoder 调用](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.cpp#L326)

## ncnn、内存和性能路线

对方固定的 ncnn 是 `f6f734f44d66f469fefee9ee401fd1cb5e3d573e`（2026-06-17），本项目固定为 `6a1bf000f363714839a36793addc8c879d3d899e`（2026-09-04）。更换依赖版本本身不是已测性能收益。

- **PE 已有缓存。** 对方的 PE 图含 26 个 SDPA cache 节点，C++ 用 26 组 `ncnn::Mat` K/V 输入输出接续历史状态。它不缺基本自回归缓存。我们验证的是较新 ncnn 的专用 allocator、容量增长和会话约定；尚未把这些接入真实 PE，更没有完成同模型提速对比。[PE 缓存管理](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.cpp#L630)、[PE 图](https://huggingface.co/wuyex/ernie-image-ncnn/blob/140a052f7919f279de7f697fa54f33bd1c0cac2b/pe/decoder.ncnn.param)
- **权重生命周期不同。** 对方一次加载包含 36 层的 `chunks` Net，并默认使用 host-memory weights；文本和 DiT 对象在后面的 VAE 阶段仍位于生成函数作用域内。本项目分阶段释放文本/DiT/VAE，DiT 每次加载一层。后者已经在本机低显存目标上运行，但会反复读取、准备和上传权重，不能直接宣称更快。[对方加载与作用域](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/ernie_image_pipeline.cpp#L1688)、[本项目层调度](../src/block_sequence.cpp)
- **GPU 边界不同。** 对方在 preprocessor、chunks、finalizer 之间通过 CPU `ncnn::Mat` 提取/传入，并在 CPU 做 Euler；本项目将这些阶段的主要激活和 Euler 保留在设备上。但当前逐层加载与 FP32 分块会增加同步，减少下载不等于已测端到端加速。
- **内存数字不可直接横比。** 对方 README 的要求是支持 ReBAR 时 RAM+VRAM 大于 32GB，或不开低内存模式/无 ReBAR 时显存至少 24GB。我们在 8GB GPU + 32GB RAM 上实际完成生成，这台机器也满足对方声明的第一类总容量条件。不能据此写成“对方必须 24GB，而我们只要 8GB”。本项目最新 FP32 长英文整卡采样峰值 4098 MiB，进程 RSS 约 5.70 GiB；还没有对方同机实测。

本项目真实新增的数值工作包括 FP16 残差保护、erf GELU 数值实现、CPU VAE GroupNorm 高精度归约、FP32 注意力补偿与有界查询工作区。它们各有局部或完整轨迹证据，但长英文/中文整体固定门限仍未全部通过。对方没有公开相同的全轨迹门限报告，不代表其作者没有测试，也不证明本项目画质更好。

## 已复现的构建文档问题

当前完整 Git tree 的根目录没有 `CMakeLists.txt`，文件在 `src/CMakeLists.txt`。README 却从根目录使用 `cmake -S .`。本轮在按固定 blob 散列核对的源码布局上执行该入口，CMake 返回 1，报告根目录不包含构建文件。

应从 `src` 选择构建源目录，并正确初始化 ncnn 子模块；本轮没有继续做完整生成器构建，不能把这处文档修正说成全平台构建已通过。公开 tree 未见项目级 CI、自动对照测试或转换脚本，GitHub Releases 列表为空；仓库内的 selftest 和 dump 入口确实存在。[README 构建说明](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/README.md)、[实际构建入口](https://github.com/futz12/ernie-image-ncnn-vulkan/blob/8dcd6e4411137d8abe92c9d78581c4c96d5182c6/src/CMakeLists.txt)

## 对本项目优先级的影响

1. 先具备处理其真实示例的能力：更长文本、非正方形尺寸、prompt 文件入口及完整 BF16 路径。
2. 接入真实 PE，验证已有原生 KV cache 方案的正确性和资源收益。图生图在 VAE encoder 独立验收后接入。
3. 同机、同权重、同初始 latent、同提示词、同尺寸/步数/精度、PE 关闭，实际运行双方并保留输出、分阶段时间和内存。双方可用的条件覆盖不足时明确报告。
4. 保留现有原始门限和负结果，同时增加提示遵循、文字生成、构图、多种子稳定性等实际成图评估。

本轮只增加比较记录与隔离验证，没有修改任一生成器的运行时代码。逐项数据、源码身份、分词器构建记录和失败配置日志见 [对照证据](../artifacts/2026-09-06/futz12-comparison/README.md)。

# 长提示词、原生 PE 与代码结构交付

日期：2026-09-06。核心实现：`9012015`。范围：Linux、ERNIE-Image-Turbo、batch=1、CFG=1、8 步，RTX 4060 Laptop 8GB / Ryzen 7745HX / 32GB RAM。

公共 C++ 生成接口、独立 CLI、按模型组件组织的运行时、2048-token 文本桶、UTF-8 prompt 文件、512×384 静态包、BF16 实验入口及完整 CPU PE 已实现。原生 PE → 文本编码 → DiT → VAE → PNG 已连接并完成实际生成。Vulkan 30/30、CPU 15/15 和 Python 66/66 回归通过。

完整质量验收仍有失败：PE 苹果图像为 24/25 张量及 PNG 通过；1080-token 中文示例 FP32 为 19/25，BF16 为 11/25。原始门槛和全部失败保留。本报告不宣称广泛质量通过、跨平台完成或优于对照项目的速度。

## 代码组织

比较了 futz12 ERNIE、nihui Z-Image 和 leejet stable-diffusion.cpp 的固定版本。采用浅层模型组件、流水线与文件 I/O 分离、公共库与应用入口分离的组织方式。三个完整 Git tree、revision、检索地址和 SHA256 保存在 [结构参考](evidence/outputs/feature-delivery-v1/structure-references/sources.json)。没有引入其运行时代码。

| 位置 | 职责与已检查的边界 |
|---|---|
| `include/ernie/pipeline.h` | 标准 C++ 类型的请求、进度、RGB 结果；没有 ncnn/PNG 私有类型 |
| `cli/` | 参数、UTF-8 文件、进度显示和 PNG；`main.cpp` 为 61 行 |
| `src/pipeline.cpp` | 333 行，连接包校验、可选 PE、文本、DiT、VAE，不解析 CLI、不保存 PNG |
| `src/prompt_enhancer.*` / `src/pe_session.*` | PE 模型和采样 / 原生不透明缓存会话与 allocator 生命周期 |
| `src/` 其他组件 | 文本、条件、去噪、分块权重、VAE、数值算子分别维护 |
| `cmake/` 和各子目录 `CMakeLists.txt` | 根构建文件 30 行；本目录拥有本目录目标，保留 `build/ernie-*` 路径 |
| `tools/` / `probes/` / `tests/` | 转换与参考 / 原生诊断入口 / 无权重回归和小型 fixture |

完整目录树、依赖方向和维护规则见 [代码结构](../../../docs/CODE-STRUCTURE.md)，转换与验收入口见 [工具说明](../../../tools/README.md)。公共 target 为 `ernie::pipeline`；当前支持源码树内使用，没有声明稳定的二进制 ABI。Vulkan 设备上下文仍由 ncnn 进程级状态管理，调用者应串行调用生成接口。

PE 权重和缓存先于文本阶段释放，文本先于 DiT 释放，DiT 先于 VAE 释放。公共接口调用测试确认无需私有头文件，链接检查确认没有 libpng。只有 PNG CLI 链接图像库。

## 新功能与组件结果

| 项目 | 实现与实际验证 |
|---|---|
| prompt 文件 | `--prompt-file` 与 `--prompt` 二选一；可选 BOM，保留空白/CRLF；非法 UTF-8、NUL、超过 1 MiB 拒绝 |
| 2048-token 文本 | 独立 pnnx 导出、完整图指纹约束；真实 1080-token 中文样本的 25 层 CPU 对照 NRMSE **7.7043e-6**，通过 |
| 512×384 静态包 | `prepare_variant.py` 独立导出文本/DiT/heads；受完整图和 FP32 重建权重散列约束复用权重；VAE 目标参考 NRMSE **9.3225e-7**，通过 |
| BF16 | 原生设备能力检查、BF16 storage、FP32 残差与 Euler；完整八步运行，质量失败仍保留 |
| PE 单块 | 真实权重、17-token 合成输入，官方完整 causal attention 对原生增量缓存；NRMSE **9.5680e-8**，通过 reset/replay、独立会话和容量检查 |
| 完整 PE | 实际 Ministral3、26 层、final norm、绑定的 embedding/LM head；CPU FP32、精确模板、greedy、temperature/top-p |
| PE 分词 | 五种输入的格式化、IDs、解码精确一致；第六种 2232-token 模板输入按容量规则拒绝；**6/6** 通过 |
| PE 完整对照 | 原始提示词 `A red apple on a wooden table.`，512×384；官方输入 139 tokens，输出 **315 tokens 到 EOS**；全部 token IDs、最终文字及 **315/315 logits** 通过 |

PE 实际采用原生 cache allocator、CPU `type=1` 句柄和 consume-and-replace。整个 315-token 运行记录 52 次 K/V 缓冲区地址变化，对应首次建立的 26 组缓存，后续地址稳定。这是地址观测，不是精确 allocator 统计或已测速度收益。prefill 逐 token 执行，输入/输出上限各 2048，缓存总容量 4096，低于官方 16384 的 query-scaling 变化边界。未实现 GPU PE 或 batched prefill。

PE 门槛在首次完整运行前固定：NRMSE ≤ 0.0002，最大差 ≤ `0.0002 + 0.0002 × reference_max_abs`。315 步中最大 NRMSE 为 **5.952998e-6**。此前输出上限 256 的对照也通过，但未到 EOS；它与完整 EOS 结果分别保留。独立重算核对了两组全部 logits、IDs、文字及 1158 个原始文件散列，见 [PE 独立复核](evidence/outputs/feature-delivery-v1/pe-independent-audit.json)。

单个 greedy fixture 不能证明更长上下文、多提示词或采样分布全面通过。相同 seed 不保证 C++ 与 PyTorch 的采样输出相同。

## 完整图像对照

每行都完成了原生 36 层 DiT × 8 步，并使用对应官方参考保存的同一份初始 latent。前三行关闭 PE；最后一行由原生端实际执行 PE，官方图像参考使用独立官方 PE 产生的文字。原生 PE 的文字和 IDs 必须精确一致。

| 运行 | 图像/文本 tokens | 精度 | 张量通过 | PNG MAE | PNG 最大差 | PNG | 整体 |
|---|---|---|---:|---:|---:|---|---|
| [结构重构回归](runs/pipeline64-structure-fp32-v1/result.json) | 64×64 / 15 | FP32 | 25/25 | 0.01139323 | 1 | 通过 | 通过 |
| [对方中文表情包提示词](runs/pipeline512x384-prompt1080-fp32-v1/result.json) | 512×384 / 1080 | FP32 | 19/25 | 0.08212619 | 17 | 失败 | 失败 |
| [同一长提示词 BF16](runs/pipeline512x384-prompt1080-bf16-v1/result.json) | 512×384 / 1080 | BF16 | 11/25 | 10.37632412 | 255 | 失败 | 失败 |
| [完整 PE 苹果图像](runs/pipeline512x384-pe-fp32-v1/result.json) | 512×384 / 315 | FP32 | 24/25 | 0.00248718 | 2 | 通过 | 失败 |

PNG 误差单位为 0..255 像素值。所有原生浮点输出到 PNG 的量化逐位通过。

固定 FP32 图像门槛：NRMSE ≤ 0.003，最大张量差 ≤ `0.0002 + 0.01 × reference_max_abs`，PNG MAE ≤ 0.1、最大差 ≤ 2。BF16 在首次运行前采用与既有 FP16 相同的门槛：NRMSE ≤ 0.15，最大张量差 ≤ `0.03 + 0.25 × reference_max_abs`，PNG MAE ≤ 12、最大差 ≤ 80。初始 latent 必须逐位一致，文本/位置条件另有固定 2e-4 门槛。任一项失败都会保留整体失败。

- PE 图像的唯一失败为 `decoded` 最大差 **0.01110547781**，超过门槛 **0.01099487681**。其 decoded NRMSE **0.00010027835** 和最终 latent NRMSE **0.00018825590** 均通过；PNG 通过不覆盖该失败。
- 1080-token FP32 失败为 `prediction-6`、`prediction-7`、`step-7`、`final`、`unpacked`、`decoded`；最终 latent NRMSE **0.00297380481** 虽通过 NRMSE 门槛，最大差仍失败。
- 1080-token BF16 的全部八个 prediction、后面三个 Euler step 及 final/unpacked/decoded 失败；最终 latent NRMSE **0.26192138466**。

原始张量、runner、PNG、参考 fixture 和每轮 Python 源码散列均经核对；封存时再次从原始 bytes 独立计算 100 项张量比较、四张 PNG 及量化判定。精确指标及失败清单见 [results.json](results.json)，每轮原始门槛在对应 `runs/` 中。没有为归档修改判定。

## 本地独立模型包与资源

| 包 | 运行文件 | 总字节 | 结果 |
|---|---:|---:|---|
| `models/turbo512x384-s2048-portable` | 136 | 23,271,871,374 | 全部实际文件、无内部链接；Python 与原生完整校验通过 |
| `models/pe-cpu-v1` | 60 | 7,680,854,847 | 完整 26 层 PE、词表和模板，无内部链接；原生/Python 校验及损坏回归通过 |

图像验收使用对应开发链接包 `models/turbo512x384-s2048-v2`；独立包逐文件复制并核对同一套运行文件。两种包的 manifest 元数据可以不同，不能仅据 manifest hash 判断运行权重是否相同。这里没有对独立包重复执行相同的长生成。

315-token PE 单独原生观测为 165.09 秒、峰值 RSS 15,420,920 KiB（约 14.71 GiB）。连接 PE 的完整图像为 533.044 秒、峰值 RSS 15,422,112 KiB（约 14.71 GiB），含模型全量散列检查和 trace。1080-token FP32/BF16 分别为 603.559 / 591.400 秒，RSS 1,746,160 / 1,858,764 KiB。这些都是单次观测，没有做对方生成器的同机受控性能对照。

两条长提示词图像运行早于 CLI/流水线拆分，重构回归和 PE 连接运行晚于拆分；每条使用各自保存的 runner 与验证脚本版本。64×64 回归期间存在 CPU 工作竞争，耗时不用于性能判断。大型 GPU 任务按顺序执行。

## 构建、测试与保留的负结果

最终 Clang Vulkan 构建、GCC CPU 构建通过；Vulkan CTest **30/30**、CPU CTest **15/15**、Python unittest **66/66**，Python 源码编译检查通过。安装后的 `ernie-image --help` 可运行。CLI-only 配置检查确认排除了 probes/tests；本次没有另行完成一个全新 CLI-only 目录的完整编译。检查汇总在 [verification.json](evidence/outputs/feature-delivery-v1/verification.json)，原始构建/测试日志在同目录。

- 第一次 variant 组装错误地依赖基线包内并不存在的转换 fixture，失败日志保留。修正为用运行时完整图/重建 FP32 权重散列验证已有 residual/BF16 包，再复用权重。
- 第一次真实 PE 单块因借用 `row_range` 的无 refcount Mat 进入 ncnn 原地路径而 SIGSEGV。会话输入边界改为 owned clone 后，真实 17-token、完整 PE 和连接图像均完成；旧失败与调试栈保留。
- 初次结构测试中，三元素 PE logits 被按含 padding 的 `Mat::total()` 误判。采样器改为验证逻辑维度，目标回归及最终全套通过；初次失败日志保留。
- 此前的 BF16 单块、FP16/FP32 长英文和中文图像失败仍在历史报告。没有远程 Actions、Windows/macOS 验收、发布或推送。

## 后续验收

优先隔离完整长文本轨迹和解码误差，扩大独立提示词/种子，随后受控测量权重读取、准备与上传，验证减少重复工作和同步的收益。PE 增加不同输入与长上下文，分别评估 batched prefill/GPU 路径。

与 futz12 相比，图生图、运行时动态尺寸、更广的默认 GPU 覆盖、更多图像格式和公开下载包仍是缺口；没有宣称所有官方功能均已移植。详细功能表见 [当前比较](../../../docs/FUTZ12-COMPARISON.md)。

本报告及小型证据由 [manifest.json](manifest.json) 固定；完整权重、runner、原始张量和生成图片留在本地 `models/`、`outputs/`，不提交 Git。原始结果和转换的历史快照不改写。

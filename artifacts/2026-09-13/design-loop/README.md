# 从设计对比到实现与回写

2026-09-13，本轮从 `codex/surpass-reference` 的 `8c3e265` 开始，ncnn 仍为 `3b7bdba7fc8aea8fd46779533eee027df77c639d`。原始输出保存在 `/var/tmp/ernie-evidence-loop-20260913-v1`；这里保存可随仓库阅读的小型记录。

工作顺序是来源核对、实现、测试、回写。[设计索引](../../../docs/DESIGN-INDEX.md)连接来源、概念和待验证项；[来源清单](../../../docs/design-sources.json)固定本轮源码身份，`python3 tools/check_design_index.py` 检查链接和源码变化。没有增加检索服务或另一套推理框架。

## 已接入：DiT 单行 mask

原来的 DiT mask 是每行完全相同的 `N×N` FP32 矩阵。现在常量只保存 `1×N`，四条 Vulkan SDPA shader 接收 `mask_h`，按实际高度取行。FP32 查询分块直接复用这一行；Q/K/V、完整可见位置、softmax、精度选择、权重放置和恢复策略没有因此改变。

方向参考 [ncnn Discussion #6998](https://github.com/Tencent/ncnn/discussions/6998)。本项目适配保留原生 shader registry 与 pipeline cache，构建目录中的 ncnn 副本承接既有 BF16 兼容处理；上游子模块不修改。原始 shader、GPU 编译单元和前一步 BF16 派生源码均需命中完整散列，见[派生身份](shader-provenance.json)。普通 cross shader 还对最后一个向量 tile 的无效 mask 读取做了边界处理。

| 总位置 N | 原 FP32 mask | 当前 FP32 mask |
|---:|---:|---:|
| 4160 | 66.015625 MiB | 16.25 KiB |
| 6144 | 144 MiB | 24 KiB |
| 10240 | 400 MiB | 40 KiB |

这是张量尺寸及实际常量构造检查，不是进程 RSS、整卡显存峰值或速度测量。CPU 的各架构 SDPA 广播支持不一致，因此在 CPU attention 调用期间兼容展开；这条 CPU 路径不承诺同样的峰值节省。

[稠密/广播测试](../../../tests/test_attention_mask.cpp)覆盖 CPU、Vulkan FP32/FP16/BF16、GQA、共享/逐头 mask、3/257 行查询、17/65 个键、连续更换 mask、重复执行，以及 124 维非 Flash 回退。每个场景都要求稠密与广播输出逐值相同且有限，FP32 另外对照独立 FP64 公式，最大绝对误差必须小于 `3e-6`。补充分支检查为 [6/6 通过](tests/ctest-fallback-v3.log)。

此前完整本地 CTest 为 [66/66 通过](tests/ctest-all-v1.log)，包括长序列 FP32 精度、受限工作区、缓存和恢复。随后新增的来源/设计检查与补充分支结果单独记录，不把旧计数当作最终矩阵。

## 已验证候选：256-token 文本图

使用锁定的官方 block 0 权重独立执行 `export_text_block.py --tokens 256`，再以 `validate_text.py --cpu-only` 对照官方输出。转换没有不支持或未转换算子；不能用改写 64/2048 图来替代这次导出。

- CPU FP32 NRMSE：`4.2257965959063374e-7`。
- 最大绝对误差：`4.410743713378906e-6`。
- 原门槛：NRMSE `2e-5`，最大差 `0.0002 + 0.0002 × reference_max_abs`。
- 结果：[CPU 对照](text256/cpu-result.json)、[官方 fixture 身份](text256/fixture.json)、[导出记录](text256/conversion.json)。独立图保存在 [测试 fixture](../../../tests/fixtures/text-s256.ncnn.param)。

随后用独立导出图认证 25 层重分桶结果，复用已逐层检查的无损权重，运行一个 87-token 的真实英文提示词。官方参考按实际长度计算，原生图填充到 256，只比较有效前缀。完整文本编码 CPU FP32 通过：NRMSE `5.977523689377264e-6`，最大绝对误差 `0.0037841796875`，参考最大绝对值 `598.9039916992188`。沿用完整文本门槛 NRMSE `0.0002` 与最大差 `0.0002 + 0.0002 × reference_max_abs`；它与上面单层的门槛和范围不同。见[25 层结果](text256/full-cpu-result.json)、[提示词及官方参考身份](text256/full-fixture.json)、[运行命令](text256/full-command.json)。

自动选择仍为已认证的 32/64/2048 来源；256 候选接下来需要 DiT 和包来源验收。1024 图像的文本容量若从 2048 降至 256，总位置会由 6144 降至 4352；这只是计划的形状变化，目前没有相应出图加速数据。

## 已验证候选：完整 26 层 PE 分块

`validate_pe.py` 新增 `--prefill-chunk` 和 `--threads`，把内部已有的分块候选接到官方逐 token/logits 检查。使用冻结的 `pe-dev-00-en` 参考：139 个输入 token，chunk 16、CPU 2 线程，生成上限 64。

- 输入 IDs、生成 IDs、增强文本均与官方一致；全部 64 组 logits 通过旧门槛。
- 最大 logits NRMSE：`6.273731256277692e-6`，门槛保持 `0.0002`。
- 52 次缓存地址变化对应 26 层首次建立 K/V，后续稳定。
- 原生进程墙钟 `44.90 s`，GNU time 最大 RSS `15,519,412 KiB`。运行受到 2 核 CPU 配额和 16 GiB cgroup 限制，主机有其他任务；这次不是配对速度测量。
- [逐 token 结果](pe-chunk16/result.json)、[原生日志](pe-chunk16/native.log)、[缓存及 EOS 状态](pe-chunk16/status.txt)。本例到输出上限，`eos=0`，不作为 EOS 覆盖。

生产默认仍为单 token 预填充。新的验证说明该候选能够通过完整模型的一例对照，不能替代其他语言、长输入和 EOS 场景。

## 模型准备与后续验证

`download_model.py --verify-with /path/to/ernie-image` 在下载散列全部通过后调用原生包校验。下载失败不启动原生程序；原生失败返回非零并保留下载结果。相关 HTTP/清单/源码测试 [28/28 通过](tests/download-source-tests.log)。公共权重地址与下载清单尚未发布。

本机另一个研究任务持续使用 GPU。本轮小型 Vulkan 算子测试已经完成，大模型 GPU 出图需要错开。64×64 CPU 完整流水线已完成，结果见下节。三平台框架 CI 使用已有工作流和已授权验证分支，结果以实际运行记录为准。

## 完整 CPU 流水线与 trace 兼容

生产二进制 `85d43bc4d15fc859090156bced134ac417797eaec05c45035e1c1fcdecfd6046` 执行原生 tokenizer、25 层文本、8 步 DiT 和 CPU VAE，使用已保存的官方参考与相同初始 latent。该 64×64 历史开发包有 272 个 DiT 文本槽，总位置 288。模型执行返回 0，所有步骤和 PNG 均已保存；[首次验收结果](pipeline64-cpu/original-result.json)在常量 mask 处失败，因为旧验收器仍按 `288×288` 读取新的 `1×288` payload。

修正 [trace 对比器](../../../tools/validate_pipeline.py)后，只重新读取同一份输出，没有重跑或替换模型结果。官方方阵 fixture、文件散列和[原门槛](pipeline64-cpu/gates.json)保持不变。仅 `constant-2` 可按单行广播，其他张量仍要求原尺寸；mask 要求广播后每个位置精确相同，不能用 `-1e30` 相对误差掩盖有效位置的错误。6 项[回归检查](tests/pipeline-trace-tests.log)覆盖旧方阵、新单行、尾行不一致、损坏值、错误形状与不完整 FP32 字节。

[重新核算结果](pipeline64-cpu/reanalysis-result.json)保留原始执行身份、全部 trace 散列、门槛身份及[复算脚本](pipeline64-cpu/recompare_pipeline.py)：**25/25 张量通过，PNG MAE `0.024495442708333332`、最大通道差 `1/255`，原生浮点转 PNG 逐值一致。** 单行 mask 的逻辑值精确等于官方方阵。

原生日志中的总耗时为 `447.792 s`，最大 RSS `1,415,500 KiB`；包括模型校验和 trace，受 2 核配额与 10 GiB cgroup 限制，不作为性能提升证据。该结果覆盖一例 CPU 完整出图；Vulkan 在本轮使用稠密/广播配对的小型网络验证。

## 三平台框架 CI

实现提交 `4e4907e750eb4f4a3f5a72239dfd7f64d333e438` 已推送至授权的 `codex/surpass-reference`。[CI 34766001397](https://github.com/mingshi2333/ernie-image-ncnn-vulkan/actions/runs/34766001397)五个作业全部成功。原始日志确认五次实际检出源码和 ncnn 修订；下表从下载的 JUnit XML 独立统计，跳过项排除在通过数之外。

| 环境 | 实际通过 | 跳过 | 失败 |
|---|---:|---:|---:|
| Linux CPU，compact reader OFF | 40 | 0 | 0 |
| Linux CPU，compact reader ON | 40 | 0 | 0 |
| Linux / Mesa 软件 Vulkan | 62 | 6 | 0 |
| macOS / MoltenVK | 62 | 6 | 0 |
| Windows / MSVC | 40 | 28 | 0 |
| 合计 | **244** | **40** | **0** |

五个作业另有各 26 项下载/清单检查通过，共 130 项。Linux Mesa 和 macOS 托管设备的 6 项跳过是原生 BF16 storage 不可用，本轮新增的 `attention_row_mask_bf16` 也按能力跳过；Windows 的 28 项跳过是没有 Vulkan 驱动。与 RAM 大小无关。Linux Vulkan 启用 Khronos validation，独立扫描无 VUID/validation error。

原始 XML、CMake 配置、测试日志和压缩作业日志保存在 [CI 目录](ci)，汇总及逐文件散列见 [summary.json](ci/summary.json)。后续 trace 读取器修正仅改变 Python 验收；本地 6 项 trace 和 8 项官方参考契约检查通过，C++/shader 与本次 CI 提交一致。文档回写不重复运行相同原生构建。

## 保留的首次失败

- [第一次配置](tests/configure.log)：适配器预期合作矩阵 Flash shader 只有一处 mask 索引，实际有两处；替换次数检查在构建前拒绝，补齐两处后继续。
- [第一次新增 mask 测试](tests/ctest-focused-v1.log)：fixture 将拥有内存的 Mat 替换成不持有内存的 channel view，释放了自身存储，造成测试内存损坏。改为直接创建二维 mask 后全部通过；这是测试构造修复。
- [第一次文本导出启动](tests/text-export256.log)：进程环境缺少用户 D-Bus 地址，systemd scope 未启动，模型没有执行。补全本机用户会话地址后在新 scope 中导出成功。

原有质量失败、BF16 精度状态与其他项目的性能报告不因本轮改变。

# P2 执行性能与内存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 P1 质量基线不退化的前提下，消除可测量的重复工作，并在 8GB 显卡、32GB 主机预算内取得端到端收益。

**Architecture:** 保留调用方管理 Vulkan command/allocator 的设计；引入有界权重会话和阶段测量。先量化读取、准备、上传、计算和同步，再逐项改动；CPU 后备实现继续可用。

**Tech Stack:** C++17、固定 ncnn/Vulkan、既有 CPU/Vulkan probes、CTest、P0 配对测量工具。

## Global Constraints

继承 [总计划](2026-09-06-surpass-reference.md)。所有新增接口与命令均为待实现目标。不得用 trace 运行计时、整卡采样冒充进程分配、取消包完整校验，或将全部展开 FP32 权重常驻 32GB 主机。已有共享 pipeline cache、设备 latent 与查询分块属于基线，不能重新算成新增收益。

## Task O1: 建立可解释的成本分解

**Files:** Create `src/execution_metrics.h`, `src/execution_metrics.cpp`, `tests/test_execution_metrics.cpp`；Extend `src/block_sequence.cpp`, `src/dit.cpp`, `src/denoiser.cpp`, `src/vae.cpp`, `src/pipeline.cpp`, `probes/block_sequence_runner.cpp`, `tools/benchmark_pipeline.py`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`。

**Interfaces:** `ExecutionMetrics` 记录 `verify/read/prepare/upload/compute/wait/download` 的 host 时间、可取得的 GPU 时间、提交数、上传/下载 bytes、live/peak allocation bytes。GPU 时间不可取得时为 unavailable，不用 host 入队时间冒充。每笔分配以 allocator/device/handle 为键；保留块、请求字节、对齐与 Vulkan 实际分配口径分别说明。

- [ ] 先写真实记账测试：分配 10、20 字节，再释放 10，再分配 15，live/peak 都应为 35；重复释放和未登记释放必须报错。测试覆盖 allocator 复用，不能把逻辑 Mat 请求峰值误作 VkDeviceMemory 峰值。
- [ ] 把现有 `BlockSequenceStats` / `DenoiseStats` 映射到统一指标，避免重复计数。权重读盘与展开/prepare 若暂时无法从固定 ncnn 分离，明确记录合并项；只在可复核的窄适配层细分。
- [ ] 在质量不变的单块和 64×64 全流程上验证开启/关闭指标输出逐位相同；误差诊断张量的下载仍只在 trace 模式发生。
- [ ] CPU RSS 由父进程/平台进程计数器获取；设备分配使用覆盖权重、blob、staging 和会话缓存的计数。对方增加相同范围的只读分配统计补丁，单独留存 diff，并验证与未插桩版本输出相同。
- [ ] 增加 `--metrics-json FILE`；这是诊断轮入口。正式速度轮关闭详细插桩，另有相同输入的内存测量轮；两边测量模式和覆盖范围一致。
- [ ] 串行获取短 512、已有长文本 512×384、PE 三种开发负载的成本分解，列出占比最高三项、理论可节省时间及资源代价。1376×768 待 P3 的原生尺寸成立后补测，不能提前假定旧转换器支持。收益上限不足以达到 S 门槛时立即记录，不能只凭提交数预言 25% 加速。
- [ ] 执行 `ctest --test-dir build -R execution_metrics --output-on-failure` 及受影响的小型回归；提交 `perf: expose bounded inference execution metrics`。

**Acceptance:** 时间范围与 bytes 可以追溯，未覆盖指标显式缺失；双方同口径峰值测量建立之后才可计算 M。

## Task O2: 有界权重会话和单块预取

**Files:** Create `src/weight_session.h`, `src/weight_session.cpp`, `tests/test_weight_session.cpp`；Extend `src/block_sequence.h`, `src/block_sequence.cpp`, `src/dit.cpp`, `src/denoiser.cpp`, `src/pipeline.cpp`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`。

**Interfaces:** `WeightBudget { host_bytes, device_bytes, prefetch_depth }`；`WeightSession::acquire(block_id)` 返回受会话管理的 lease；`release_after(lease, completion)` 只有已完成的 command 才可回收相关权重。`cancel()` 必须 join worker、等待必要设备完成并释放资源。开始只允许 `prefetch_depth=0/1`，租约不可复制成独立所有者。

- [ ] 用 fake loader 和可控 completion 写状态测试：当前块在飞行中时不得释放；下一块准备占用计入预算；超预算等待/降到同步流式；读盘失败、取消和销毁都不能泄漏线程或分配。
- [ ] 先将已有 `Stream` 行为装入 depth=0，保持顺序与数值，验证单块、36 块和历史完整开发 fixtures。保留既有显式 `Resident` probe 作实验，不能作为默认的无限内存策略。
- [ ] 依据 O1 判断能否复用解析、原始 BF16 映射或准备结果。Linux mmap 与 Windows 文件映射必须有同样的只读所有权和错误语义；页缓存/RSS仍计入测量，不把磁盘映射称为零内存。
- [ ] 只让一个后台 worker 做受预算约束的文件读取/CPU 准备；所有未被固定 ncnn 证实可并行的 Vulkan 对象创建与上传仍由所有者线程执行。禁止 worker 与主线程对同一个 Net 并发修改。
- [ ] 优先评估复用每步重复读取的 head/小型条件权重，再测单块预取；不假定 36 个展开权重同时缓存有益。每个策略输出实际命中、节省的读/prepare时间及新增驻留字节。
- [ ] 固定输入分别测 stream、depth=1、预算内部分常驻；质量、端到端、RSS、设备峰值共同决定默认值。触发 swap/OOM 的方案记录失败并撤回默认。
- [ ] 执行 `ctest --test-dir build -R 'weight_session|pipeline_api' --output-on-failure`，串行运行受影响的 8 条开发集；提交 `perf: reuse bounded model state and prefetch streamed weights`。

**Acceptance:** 有界会话在正常完成、加载失败和取消下都安全；优化相对本项目 P1 基线有重复可见的净收益。局部无收益的分支不因计划已列出而强行保留。

## Task O3: 减少重复条件计算与 attention 同步

**Files:** Extend `src/conditioning.h`, `src/conditioning.cpp`, `src/dit.cpp`, `src/denoiser.cpp`, `src/ernie_attention.cpp`, `probes/attention_workspace_probe.cpp`, `tools/prepare_block.py`；Create `tests/test_conditioning_reuse.cpp`，必要时更新既有 shader hash 契约和对应来源记录。

**Interfaces:** 请求内固定条件与每步时间条件由不同类型表示；固定条件的身份包含模型、尺寸、token IDs 与输入特征 hash。其生命周期仅覆盖本次请求，不引入隐式跨请求全局缓存。

- [ ] 画出实际 head/block 数据依赖，再标注可复用节点：原始文本投影只有被证明与 timestep/latent 无关才可请求内复用；shared modulation 只在相应 timestep 内复用。DiT 中间文本 hidden states 随联合 attention 改变，禁止跨去噪步缓存它们或其 K/V。
- [ ] 用两个不同 timestep、两份不同 latent、同一文本的真实激活对照，检验被标为固定的节点；增加模型/形状变更必失效的测试。未通过依赖证明的节点保持现状。
- [ ] 在不覆盖仍被 GPU 读取的 scratch 的前提下，评估多查询块共用一次提交、受预算约束的 scratch ring 或减少 host 等待；保留所有 K/V、softmax/P@V 补偿语义与 FP32 residual。
- [ ] 先以真实 4160-token Q/K/V 检查与已验收分块实现逐位相同；改变归约次序的候选必须重新通过固定 FP64 算子门槛与完整轨迹门槛，不能只用小张量平均误差验收。
- [ ] 测长文本 + 大图的最大工作区，检查 128 query 分块边界、尾块、空/短文本和有限值报告。出现超预算时选择已验证的小块策略；选择规则仅依据形状/预算。
- [ ] 执行 `ctest --test-dir build -R 'conditioning_reuse|attention_' --output-on-failure`；以独立 commits 分别提交条件复用和同步变更，每个 commit 附自身完整开发回归及配对计时。

**Acceptance:** 复用由数据依赖证明；更少提交带来实际端到端收益且不扩大内存超过预算。若只是把等待挪到其他位置，报告无收益。

## Task O4: GPU 文本/VAE 与 PE prefill

**Files:** Extend `src/text_encoder.cpp`, `src/vae.cpp`, `src/prompt_enhancer.cpp`, `src/pe_session.h`, `src/pe_session.cpp`, `probes/pe_block_runner.cpp`, `tools/validate_text.py`, `tools/validate_dit_heads.py`（已有 `component=vae`）, `tools/validate_pe.py`；Create `tools/profile_device_policy.py`, `tests/test_device_policy.cpp`。

**Interfaces:** 公开设备策略由 P3 的 `GenerationRequest` 扩展承载；内部 CPU/Vulkan 各阶段返回相同张量契约。PE 增加 `prefill(valid_embeddings, valid_positions)`，不把 padding 写入 cache；返回末尾真实 token logits 与实际 position。

- [ ] 核对现有 GPU 文本/VAE 入口能否复用。先单层/小图验证，再独立大图；文本保持官方 block 24 输出、原生 mask 和 YaRN，VAE 保持已验证归一化语义。
- [ ] GPU 文本对 32/64/2048 文本桶及真实有效长度通过固定 conditioning 门槛，随后接入完整开发集。GPU VAE 对同一保存 latent 通过独立 VAE 和最终像素门槛，再做连接回归。
- [ ] 精度不通过时明确选择 CPU 后备并显示实际执行设备；用户强制 GPU 且不可满足要求时给出明确错误。不能接受参数后悄悄忽略设备选择。
- [ ] 将 PE prefill 与逐 token 基准比较：真实 token 长度为 1、31、32、33、127、128、129、2048，末 token logits、position、后续 decode/reset/容量行为都一致。选固定 chunk 大小须来自性能和容量测量。
- [ ] 使用真实 `PeSession` 测 prefill，不在 Python 中重写缓存当作原生测试。greedy 315-token 原 fixture 和 Q3 的 12 条 PE 输入保持通过。
- [ ] 比较 stage-level 与完整生成时间；PE 整体耗时、cache/权重峰值、文本/VAE设备策略在报告中独立显示。GPU PE 不是本阶段硬依赖：对方 PE 也是 CPU；只有 CPU prefill 与整体验收完成后才开独立 GPU PE 实验。
- [ ] 串行执行对应验证器，再跑 P0 `--suite performance --development`（该模式仅报告调优，不写最终胜出）；P3 尚未实现的图生图 case 记 incomplete，待 F2 后补全，不能从正式六项分母删去。提交 `perf: validate stage device policies and batched PE prefill`。

**Acceptance:** 默认设备策略只使用通过正确性和资源验收的路径。P2 结束报告 S/M 目标的开发测量进度，正式胜出仍等待 P5。

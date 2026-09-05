# 后续组件与多 block 验证

这些命令基于 [单 block 复现](REPRODUCE-BLOCK.md) 的环境和模型目录。当前组件运行并不等于完整文生图，所有参考输入范围和失败项都需随结果保留。

## 官方权重的直接转换

已核验的三个静态桶计算图，去掉 token 数量参数后有相同的完整图 SHA-256。`build_dit_weights.py` 要求这个图指纹匹配，再将官方 11 个 BF16 张量按固定映射写入 ncnn 权重流。七个 Gemm 使用原生 BF16 段，四个 RMSNorm 使用 FP32 affine 段。

```sh
.venv/bin/python tools/fetch_component.py --block 1
.venv/bin/python tools/export_dit_block.py --block 1 --fixture-only --height 4 --width 4 --text-tokens 272 --valid-text 270 --output models/reference/block1-288
.venv/bin/python tools/build_dit_weights.py --template models/converted/block0-long-text/runtime --weights models/official/dit-block-01.safetensors --fixture models/reference/block1-288 --output models/converted/block1-288
.venv/bin/python tools/run_block_matrix.py --model models/converted/block1-288 --output outputs/block1-288
```

第 0 块直接写出的文件与 pnnx 导出后 BF16 打包的文件逐字节相同，大小均为 436,241,436 字节。这条路线省去相同拓扑的重复 tracing，以及中间的 FP32 权重文件。它仍需要官方参考和各块的数值验证。

顺序处理明确指定的 block 范围：

```sh
.venv/bin/python tools/convert_dit_blocks.py --template models/converted/block0-long-text/runtime --start 0 --end 36 --output models/dit-288
```

该流程逐块下载、生成官方参考、转换并运行四种 CPU/Vulkan 配置。执行文件先复制成不可被开发构建替换的运行快照。日志保留在每块目录，整体状态保留在 `conversion-run.json`。遇到失败就停止，不将已经运行过的部分标为完整模型通过。

需要完成剩余块的诊断时，可以显式使用 `--keep-going-on-numerical-failure`。只有四个执行器均正常退出、但数值门槛失败时才继续；下载、转换、崩溃和超时仍会停止。失败门槛原样保留，整体退出码仍为非零。第 31 号 block 的 BF16 独立合成输入测试就是一个保留的失败项，不能根据后续串联结果删除它。

下载采用严格校验长度的 32MiB HTTP Range 窗口，单窗口最多尝试三次。失败的组件 `.partial` 文件需要另存后再重试，不能当作完成的 safetensors 文件。默认完整下载和转换约需要两份 DiT 权重的磁盘空间，已有组件可验证后复用。

## 多 block 权重生命周期

```sh
.venv/bin/python tools/validate_block_sequence.py --model models/converted/block0-long-text/runtime --model models/converted/block1-288 --output outputs/sequence-01
```

验证器先生成串联官方 FP32 参考，再运行 CPU FP32、Vulkan FP32/FP16/BF16。一块权重完成执行后，stream 模式释放该块的 Net，激活保留在调用方的 VkMat allocator 中。block 之间没有下载激活。resident 对照最多保留两块，避免将约 15GiB 的全部权重直接加载到 8GB 显卡。

`peak_loaded_nets` 是执行逻辑约束的同时存活 Net 数量，不是显存测量。各次加载、计算时间和进程最大 RSS 单独记录。模型输出需要与官方参考比较，相同精度下 stream/resident 输出还必须逐位一致。

FP16 RMSNorm 大输入的平方溢出问题及项目修复见 [专门复现](NCNN-RMSNORM-REPRO.md)。现在项目默认注册 FP32 临时计算的 RMSNorm 覆盖层，不能将修复前后的低精度结果混为一个版本。

## DiT 输入、时间条件、输出层与一次预测

```sh
.venv/bin/python tools/export_dit_heads.py --download --output models/dit-heads-288
.venv/bin/python tools/validate_dit_heads.py --model models/dit-heads-288/input --output outputs/input-head
.venv/bin/python tools/validate_dit_heads.py --model models/dit-heads-288/output --output outputs/output-head
.venv/bin/python tools/validate_dit.py --input-head models/dit-heads-288/input --output-head models/dit-heads-288/output --model models/converted/block0-long-text/runtime --model models/converted/block1-288 --output outputs/dit-prefix-two
```

最后一条命令明确测试两层前缀。完整 DiT 检查必须按顺序传入第 0 至 35 层的 36 个 `--model` 参数。验证器拒绝不连续索引、不同静态 token 数、不同前后处理尺寸以及权重/checksum 不匹配。`--timestep` 默认 1000；参考与运行时读取同一份时间特征、图像 latent、合成文本 embeddings、RoPE 和 mask。

输入层含 1×1 图像投影、3072→4096 文本投影、时间 MLP 和六个 shared AdaLN 向量。输出层保留无 SiLU 的 conditioning linear、非 affine LayerNorm、image-token slice 和 128 通道恢复。导出 wrapper 先与官方组件输出逐位核对，随后才运行 ncnn。第一次 finalizer 导出曾出现 batch 轴告警，失败产物保留；当前表达全程保持 batch=1。

运行路径为输入层 → 一次加载一个 block → 输出层。GPU 中间激活留在设备上，六个调制向量只去掉经检查的 singleton 维度，保留显式 allocator 生命周期。卷积的精简构建还必须启用其内部 Padding 依赖。LayerNorm 使用与 RMSNorm 相同的 FP32 临时计算策略。

这项测试是一轮 DiT 预测；默认 4×4 packed-latent 网格、272 个文本位置，其中两个为 padding。文本 embeddings 为合成输入，时间正弦特征由官方参考预先计算。它不覆盖真实文本编码、C++ 时间特征生成、8 步 Euler 闭环、VAE 或 PNG。小尺寸预测成功也不能代替 1024×1024 的全模型测试。

## Vulkan pipeline cache 对照

序列执行程序默认在调用方持有一个 `ncnn::PipelineCache`，通过 `Option::pipeline_cache` 传给每个 Net，直到所有设备命令完成才销毁。这样可跨 block 和前后处理层复用已经创建的 pipeline。`--isolated-pipeline-cache` 恢复每个 Net 自己持有 cache 的旧行为；CPU 不使用这个选项。

先用 `validate_block_sequence.py` 或 `validate_dit.py` 的 `--isolated-pipeline-cache` 选项生成旧行为的完整 baseline，再运行：

```sh
.venv/bin/python tools/compare_pipeline_cache.py --baseline outputs/sequence-36-baseline --output outputs/cache-sequence-36
.venv/bin/python tools/compare_pipeline_cache.py --baseline outputs/dit-36-baseline --output outputs/cache-dit-36
```

对照工具要求 baseline 的四种配置均执行完成，并检查原输入、模型文件与 fixture 的 checksum。它分别重跑三种 Vulkan 精度，要求新的输出散列与对应 baseline 完全一致。统计分离加载/初始化与计算，加载阶段也包含 ncnn 权重准备，不是纯磁盘吞吐。一次前后对照不能宣称完整图片生成提速。

这是 Vulkan pipeline 编译缓存，与自回归模型的 K/V cache 是不同机制。它不会复用 DiT 的跨去噪步 K/V，也没有将 36 块权重同时保留在显存。

## Euler 和 VAE 前处理

```sh
.venv/bin/python tools/validate_latent_ops.py --output outputs/latent-operations
```

当前实现支持 batch=1、FP32 latent、shift=4 的确定性 FlowMatch Euler。7 套 fixture 覆盖 1/7/8/17/50 步、非方形和 1024 分辨率对应的 latent。官方 sigma/timestep 必须逐位一致，每一步 latent、最终 BN 反归一化和 128→32 通道解包分别对照。GPU 有逐步提交和集中提交两种检查。

参考使用合成的逐步预测与 BN 统计，包含零方差及小方差以检查 `1e-5` epsilon。它没有调用真实 DiT 或 VAE decoder。FP16/BF16 scheduler 的舍入顺序尚未实现，不能把 FP32 验证外推为低精度 scheduler 一致。

## 原生 tokenizer

Tokenizer 通过 C++ RAII 接口调用官方 Tokenizers 0.22.2 的静态 Rust 库，与参考环境版本一致。编码不依赖 Python，也不访问网络。为了不强制所有探针构建都依赖 Cargo，该组件默认不构建。

```sh
.venv/bin/python tools/fetch_tokenizer.py
cmake -S . -B build-tokenizer -DERNIE_BUILD_TOKENIZER=ON -DERNIE_ENABLE_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-tokenizer --target ernie-tokenize -j 4
.venv/bin/python tools/validate_tokenizer.py --output outputs/tokenizer-validation
build-tokenizer/ernie-tokenize models/tokenizer prompt.txt
```

`tokenizer/Cargo.lock` 固定传递依赖和校验值。构建使用 `--locked`。当前 C++ 层完整保留 UTF-8 输入，包括文本文件中的 NUL 字节，非法 UTF-8 明确拒绝。官方 JSON 决定 ByteLevel/BPE/special-token 行为，调用显式启用 special tokens，使用 2048 token 右截断，不 padding。

已验证的 48 个样本含多语种、Unicode 组合字符、emoji、BOS/special tokens、空文本和长文本。`ignore_merges=true` 配置被保留，当前候选扫描未找到开关前后不同的真实词表反例，不能宣称已经建立该开关的差异测试覆盖。Tokenizer 通过不代表 26 层 Mistral3 文本编码器已移植。

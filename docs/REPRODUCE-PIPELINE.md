# 完整原生 pipeline 复现

环境沿用 [单 block 复现](REPRODUCE-BLOCK.md)，并按 [README](../README.md) 启用 tokenizer、生成器和 libpng。Python 用于转换/参考，`build/ernie-image` 本身是原生程序。以下示例使用一致的新目录命名；本机历史验证包为 `models/pipeline64-residual-v1` 和 `models/pipeline1024-residual-v1`，底层块来自多次独立转换。

所有输出目录应是新路径，失败结果另存。一次只运行一个大 GPU 作业。已用主机为 8GB 显卡、32GB RAM；完整 1024 运行峰值 RSS 约 23.0GiB，系统 swap 增加约 0.96GiB。不要在内存压力下重复整图高分辨率 VAE 的 pnnx 转换。

## 1. DiT 基础权重

先按单 block 文档生成并验证 `models/converted/block0-long-text/runtime`（288 tokens）与 `models/converted/block0-1024/runtime`（4160 tokens）。再按 [组件复现](REPRODUCE-COMPONENTS.md) 转换完整 36 层：

```sh
.venv/bin/python tools/convert_dit_blocks.py \
  --template models/converted/block0-long-text/runtime \
  --start 0 --end 36 --keep-going-on-numerical-failure --output models/dit-288
```

该诊断选项让数值失败后继续检查剩余层，不把失败计为通过。已知 BF16 独立块 31/33/35 失败，因此整体命令会返回非零；需要逐项确认全部执行完成以及 CPU FP32、Vulkan FP32/FP16 的门槛。下载、转换、崩溃或超时仍会停止。本原型生成使用 FP16，不启用 BF16。

## 2. 文本路径

```sh
.venv/bin/python tools/fetch_tokenizer.py
.venv/bin/python tools/fetch_pipeline_components.py
.venv/bin/python tools/audit_text_selection.py --output outputs/text-selection
.venv/bin/python tools/export_text_block.py --tokens 32 --output models/text-template-32
.venv/bin/python tools/build_text_weights.py --template models/text-template-32 \
  --output models/text-32 --embedding
```

`audit_text_selection.py` 用完整小尺寸 Mistral3 模型和层钩子验证 hidden-state 选择，不把小模型检查称作真实权重验证。真实路径使用官方 BF16 权重的前 25 层，并与实际长度的官方 Mistral 参考比较；ncnn 使用 32-token 静态桶和因果 mask。

```python
# 从项目根目录保存或执行以下 Python；它顺序启动一个验证作业。
import subprocess
cmd = ['.venv/bin/python', 'tools/validate_text.py',
       '--embedding-package', 'models/text-32',
       '--prompt', 'A red apple on a wooden table, soft daylight, realistic photo.',
       '--output', 'outputs/text-real-prompt']
for i in range(25):
    cmd += ['--model', f'models/text-32/block-{i:02d}']
subprocess.run(cmd, check=True)
```

32 tokens 包含 BOS。`ignore_merges=true` 保留在 tokenizer 配置中，但当前没有该开关的真实词表差异反例。`hidden_states[-2]` 为 block 24 输出，无 final norm；不导出第 26 层、视觉模型或 LM head。

## 3. VAE

先导出和检查小尺寸完整图，再仅执行大尺寸官方参考，并特化两处已审查 reshape：

```sh
.venv/bin/python tools/export_vae.py --height 8 --width 8 --output models/vae-8
.venv/bin/python tools/validate_dit_heads.py --model models/vae-8 --output outputs/vae-8
.venv/bin/python tools/export_vae.py --height 128 --width 128 --reference-only \
  --output models/vae-128-reference
.venv/bin/python tools/specialize_vae.py --template models/vae-8 \
  --reference models/vae-128-reference --output models/vae-128
.venv/bin/python tools/validate_dit_heads.py --model models/vae-128 \
  --cpu-only --output outputs/vae-128-cpu
```

8×8 / 128×128 是解包后的 32 通道 VAE 输入，分别输出 64×64 / 1024×1024。转换器严格匹配整个小尺寸图的 SHA-256，仅修改 attention 前后 flatten/unflatten；卷积、归一化、最近邻上采样和权重不变。大尺寸官方 fixture 的来源、权重、形状与张量散列必须匹配，再运行完整输出数值门槛。

原始大尺寸整图 pnnx 导出因内存持续增长而停止。CPU 原生 GroupNorm 的大尺寸归约曾两次超过固定 `2e-5` NRMSE 门槛；仅关闭 Winograd 没有修复。当前 CPU GroupNorm 使用 FP64 均值和中心方差归约，FP32 激活与 affine 运算，NRMSE 2.02e-6。GPU VAE 仍使用原生 GroupNorm，尚未通过完整 1024 图像生成验收；默认选择 CPU。

## 4. 前后处理、FP32 残差与静态桶

```sh
.venv/bin/python tools/export_dit_heads.py --download --height 4 --width 4 \
  --text-tokens 272 --output models/dit-heads-288
.venv/bin/python tools/export_dit_heads.py --download --height 64 --width 64 \
  --text-tokens 64 --output models/dit-heads-4160
```

每个输入/输出层可分别用 `validate_dit_heads.py --model DIR --output NEWDIR` 复核。然后复用已检查的形状无关权重，使用各桶独立导出的计算图：

```python
import subprocess
for tokens, template in [(288, 'models/converted/block0-long-text/runtime'),
                         (4160, 'models/converted/block0-1024/runtime')]:
    cmd = ['.venv/bin/python', 'tools/rebucket_dit.py', '--template', template,
           '--fp32-residual', '--output', f'models/dit-{tokens}-residual']
    for i in range(36):
        cmd += ['--model', f'models/dit-288/block-{i:02d}/runtime']
    subprocess.run(cmd, check=True)
```

`--fp32-residual` 是真实文本 FP16 生成所需：两处 ADD 保持 FP32 skip，归一化后的投影输入回到模型存储精度。省略该选项的旧图会在真实激活超过 65504 时溢出。新桶 manifest 明确没有该桶的每块独立数值 fixture，不复制旧桶参考冒充新桶通过；闭环验证单独执行。

## 5. 组装本地模型包

```python
import subprocess
for pixels, tokens, vae in [(64, 288, 'models/vae-8'),
                           (1024, 4160, 'models/vae-128')]:
    heads = f'models/dit-heads-{tokens}'
    cmd = ['.venv/bin/python', 'tools/prepare_pipeline.py',
           '--input-head', heads + '/input', '--output-head', heads + '/output',
           '--vae', vae, '--embedding-package', 'models/text-32',
           '--output', f'models/pipeline-{pixels}']
    for i in range(36):
        cmd += ['--dit', f'models/dit-{tokens}-residual/block-{i:02d}']
    for i in range(25):
        cmd += ['--text', f'models/text-32/block-{i:02d}']
    subprocess.run(cmd, check=True)
```

组装器核验 tokenizer 版本/文件、层顺序、静态 shape、组件及权重散列，写入独立 manifest、RoPE 和官方 BN 统计。BN 使用实际 pipeline 的 `eps=1e-5`。当前包依赖本地符号链接，不能单独搬走。原生 CLI 读取配置和张量，完整文件散列检查由组装/验证工具负责，独立分发及原生 manifest 校验留待后续。

## 6. 完整对照和本机资源测量

```sh
.venv/bin/python tools/validate_pipeline.py --model models/pipeline-64 \
  --precision fp16 --output outputs/full-64
.venv/bin/python tools/benchmark_pipeline.py --model models/pipeline-1024 \
  --precision fp16 --vae-device cpu --output outputs/full-1024
```

第一项先分阶段执行固定版本官方模块，再使用保存的同一份初始 latent 运行原生程序。FP16 门槛在执行前保存：各去噪/最终张量 NRMSE ≤ 0.15，同时满足 `0.03 + 0.25 * max(abs(reference))` 最大绝对误差；PNG MAE ≤ 12/255、最大差 ≤ 80/255。文本/位置条件有独立更严门槛。逐步 latent、预测、解包、解码和 PNG 量化全部单独核对，任一失败返回非零。门槛是工程数值标准，不等同于感知质量评估。`--reference COMPLETE_DIR` 仅复用配置、prompt、步数和所有散列一致的完整参考。

第二项只测试 1024 原生功能与资源，结果明示 `quality_validated=false`。它保留 binary 快照、来源和包散列、实际随机 latent、逐步张量、PNG 散列、GNU time 最大 RSS、100 ms 整卡显存采样及系统内存/swap 前后状态。时延包含 trace 的保存开销；显存样本含其他程序且可能漏过短峰值，不是精确本进程计量。

本次 1024 输出为木桌上的红苹果，符合所用单个提示词的肉眼检查。完整高分辨率官方去噪对照、多提示词质量、长文本、重复性能与不同设备仍需单独评估，不能从这一张图外推。

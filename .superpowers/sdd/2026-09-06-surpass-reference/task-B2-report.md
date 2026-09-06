# Task B2 部分交付（2026-09-06）

状态：**适配器、独立构建和部分身份审计完成；完整对方成图基线尚未执行。B2 acceptance 未完成，S 不能关闭。** 下载继续运行，交给 root 同一进程监督。没有运行任何大型 peer 推理或 GPU 工作。

## 已执行

- 新建 `tools/port_adapters.py`、`tools/compare_ports.py`、`tools/audit_port_weights.py`、`tests/test_port_adapters.py`，只修改本任务文件。
- 真实 Git checkout 固定 `8dcd6e4411137d8abe92c9d78581c4c96d5182c6`；递归子模块保持其 ncnn `f6f734f44d66f469fefee9ee401fd1cb5e3d573e`。从 `src/CMakeLists.txt` 用 Clang 22.1.8、CMake 4.3.0、最多 2 jobs 编译成功。源码状态 clean，**无平台补丁、无本方数值补丁**。
- 对方可执行文件 SHA256 `49422b21635868cd3d4487b80d9c5dd1e421be8c9462a85cc03dc837e5ed82ac`。源码、子模块、构建参数及完整日志见 `outputs/reference-port-v1/build-provenance.json` / `logs/`。
- 合成唯一编号 CHW/HWC 映射测试按元素位置验证，错误解释布局得到不同 hash。另编译独立小型 CPU layout probe，直接包含未修改 peer pipeline 源码，调用其真实 load/dump 函数，把历史完整生成的 64×64 initial dump 往返；两侧 SHA256 均为 `4b6fbf045688cc0ed9c9e589b9dc0cd2955f2ac9847a581357c4fc16d4c204e1`。这证明实际文件布局与 bytes 往返，**不是完整模型生成**。两侧原生 patchified latent 均 CHW。
- 8 项 unittest 通过：布局、错误尺寸/非有限数、配对缺失/不匹配拒绝、prompt 换行保留、精度/线程能力拒绝、流式 BF16/矩阵转置、未知层停止、GEMM 转置及多余 bytes、日志观测（部分项目同一测试内）。Python 编译检查通过。初始 red 测试阶段未单独保存；不冒称按完整 red/green 顺序执行。
- 直接使用 B1 原 manifest，通过 `verify_inputs` 校验，8 项 calibration 在未提供二进制配置时全部保留 incomplete；`--development` 不产生胜出。执行证据 `calibration-incomplete-v3/result.json`。`--ports FILE` 独立配置不修改冻结 corpus。

## 合同和明确限制

- Peer `--prompt-file` 去掉末尾 CR/LF；适配器用 literal `--prompt` argv 保存原始文本。输入显式 saved FP32 CHW，检查尺寸/hash；trace 时检查实际 initial canonical SHA。
- Peer `--fp32-storage` 或默认 BF16，源码显式禁用 FP16。Candidate 固定 CPU threads=4，因此 4 可用、8 unavailable；peer grid 4/8。low-vram off 必须有显存容量预检，默认 unavailable，不能用故意 OOM 代表容量结果。
- Peer greedy PE 通过源码确认 `temperature <= 0` 为 argmax；映射 `--pe-temperature 0`，candidate 映射 `--pe-greedy`，其余 max tokens/top-p/seed 显式。PE 原生文本输出一致性还没有执行证明。
- 每阶段 dtype/device 分别记录；peer 精度影响 text/DiT/VAE，candidate 精度入口主要影响 DiT；FP32 residual/Euler 与 BF16 文件存储不能混称全部低精度运算。精度不一致不能形成正式数值配对。设备允许按实现不同，但必须显式记录。
- Runner 先核对完整模型包/peer 固定资产 hashes，保存 runner snapshot 和 launch.json，再使用外部 monotonic 时钟与 `/usr/bin/time -v` 保存完整 stdout/stderr/exit/resource/output hashes。超时杀整个 process group，避免遗留 GPU 子进程。**此 runner 的实际大型进程路径尚未执行**。
- 运行成功仍 `quality_status=incomplete`；没有把 PNG 文件存在伪装为完整质量通过。质量验证、正式 AB/BA 五对调度、正式聚合由 B3/后续阶段完成；工具不复制 B3 聚合策略。

## 权重身份：部分真实 bytes 审计

固定 HF primary endpoint `wuyex/ernie-image-ncnn@140a052f7919f279de7f697fa54f33bd1c0cac2b` 的 API 带 LFS SHA/尺寸元数据已保存。下载逐文件验证大小与 LFS SHA256，完整资产约 61 GB；PE embedding/head 相同 SHA 可本地 hardlink 复用。

`weight-audit-v2/audit.json` 已实际完成：1015 个官方逻辑张量规范化流散列、当前已下载 peer 的 119 个权重/常量记录，其中 **27 个内容 hash 匹配、92 个未匹配**。匹配分布：preprocessor 9、finalizer 2、text embedding 1、已解析 VAE decoder 前缀 15。不是全模型同权重证明。

缺口保留：DiT chunks、text encoder、PE 大文件尚缺；VAE `MultiHeadAttention` 解析器尚未实现，遇到后立即停止该文件；VAE encoder 官方权重未在当前官方本地组件库存，故许多值未匹配；finalizer 两组 GEMM 和衍生常量需要进一步逻辑/融合映射。内容 hash 匹配也不独立证明图中逻辑连线相同。报告固定 `status=unproven` / `product_comparison_only` / `allowed_to_close_S=false`。

审计每块最多 1,048,576 元素，FP32/BF16/F16/I64 规范化为 `<f4`，无多 GB 物化数组；官方原文件也按已有 manifest 完整校验。首轮因 VAE BN 的 I64 scalar 未支持而失败日志保留，第二轮补充 I64 后完成。之后对非转置路径改为实际文件块流读取，数值合同不变、合成回归通过；第二轮运行的是修改前相同规范化算法，RSS 观测约 37–49 MiB。转置以 mmap 列视图分块，未复制完整矩阵。

## 仍在运行与后续步骤

- 活跃下载 unified-exec **session 97237**，PID **1524331**。精确 `/proc` start ticks、cmdline、观测时间、partial 文件大小与当时所有已验证资产 SHA 在 `outputs/reference-port-v1/download-handoff.json`。日志 `logs/download-v2.log`；进度 `assets-manifest.partial.json`；完成时 rename 为 `assets-manifest.json`。旧下载 session61230 已明确停止，535+ MiB PE partial 保留；v2 有 Range resume/retry，不能仅因单次 wait 超时重启。
- 已完成审计 session23955；旧首轮 session44716 已失败并保留。独立编译 session33612 已完成；layout helper session85928 已完成。
- 可审查的首个64×64命令、原始输入及配置在 `first64-plan.json` / `first64-inputs/`。这是单独 development smoke，原 B1 frozen development 没有64×64，因此没有修改其样本集合。**未启动**。
- Peer 非 PE 活跃模块文件总大小 45,582,873,100 bytes，实际 RSS 尚未测。系统31GiB物理RAM与62GiB zram；zram不是独立磁盘容量，不能直接相加证明可容纳。正式启动前由 root 再预检，等 Q2 结束并明确分配 GPU 槽。
- 下载完成后：重做全部 peer 流审计，补 graph/derived weight 映射；预检后执行64×64、512×512，再1024/长文本/PE。完整图像及端到端共同质量/校准仍全部 pending，失败保留分母，不能据缺记录计算速度收益。

未 push、发布或上传。所有 weights/build/大证据保留 outputs，不提交 Git。

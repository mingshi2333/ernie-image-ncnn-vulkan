# Q2 实际中文 native step6 重放准备与执行

本报告前文保留运行前准备记录；已授权执行结果见末节。目标是先验证六个实际输入能否由固定单步runner重现历史native prediction-6，不是官方数值验收；无官方gates，不将重放成功称作修复中文22/25失败。协议名 native-chinese-step6-replay-v1，以exact FP32 output bytes作为重现判据，同时保留不相等时的NRMSE/max/不同float数作为诊断信息。

## 实际输入身份

来源 `outputs/pipeline1024-chinese-vectordown-fp32-v1`，源runner SHA `7d9ea0d6dd6633781f17671bc5c4fa2f8fbf01d5b88d6e2c114e21175168b06c`，原始运行是saved-candidate text诊断。该runner实际文件SHA已经重新核对。25项comparisons无重复，以下每份trace SHA均匹配源result对应记录，shape/byte size/finite亦检查。

| 单步输入 | 原始来源 | 连续FP32布局 |
|---|---|---|
| in0 | trace/step-5.f32 | CHW [128,64,64] |
| in1 | trace/padded-text.f32 | [64,3072] |
| in2 | 已独立验证native-direct.f32 | [4096] |
| in3 | trace/constant-0.f32 | [4160,128] |
| in4 | trace/constant-1.f32 | [4160,128] |
| in5 | trace/constant-2.f32 | [4160,4160] |

expected 使用原生trace/prediction-6.f32，SHA `622738cfeb9f811e162ac0c8053ae91917d316a1513856044a8f72539f1f3fe7`，不是官方prediction。native feature SHA `5919ea0805bf18fa0de6fcb1ebf080cd7e62af72ec411a57f5f9dce71cd604c1`，来自已检查直接production函数与f16a...denoise runner两种decimal路径的同一输出；step6 timestep `571.4285888671875`。

原生constant与官方的小tensor对比：

| 原生数据 | 逐位相同 | max abs | NRMSE |
|---|---|---:|---:|
| padded-text | 否 | .00091552734375 | 1.902328094e-6 |
| RoPE constant-0 | 否 | 5.960464478e-8 | 2.445645231e-8 |
| RoPE constant-1 | 否 | 5.960464478e-8 | 1.457796229e-8 |
| mask constant-2 | 是 | 0 | 0 |

因此不能在“真实native调用重放”里沿用先前teacher-force的官方RoPE。此次全部换回原始native保存常量。差异大小本身不证明贡献或主因，需先看重放结果。

## 源和执行选项核查

预测器仍为a4a80b...冻结block-sequence runner，与上一单因素实验相同。固定package SHA72bb195a...、36 DiT blocks及两个heads、Vulkan FP32、stream、shared PipelineCache、host_weights=false、默认GPU、默认4线程请求，两物理核affinity控制保持。

历史pipeline blob6876a32与a4a...执行快照的pipeline使用相同FP32选项：fp16/bf16 storage/packed/arithmetic关闭；CPU base option未关闭SGEMM/Winograd，与block runner相同。当前并行F2中的pipeline有关闭卷积优化的新设置，不能倒推为历史7d9...设置。初读活动源码产生的选项疑问已在核对历史blob后纠正。

历史完整denoise保持FP32 pack1 master latent，再无损convert_packing到pack4喂DiT；单步runner读取同一CHW bytes并按FP32 option上传打包。width/height64使通道面连续且对齐；text、time、RoPE、mask shape对应相同ncnn输入。没有在preparation里做数值转换或重排实际bytes。

Git d9f3b4b→27714e0之间 conditioning、denoiser、latent_ops、GELU、attention、RMSNorm、residual、attention shader配置无diff；DiT/block改动为已审查ComponentFiles内存param加载桥接及对应重载。历史7d9...worker主要封存Python工具，并非完整native C++构建输入清单；本报告以已审查历史pipeline blob、Git范围与实际runner SHA做可追溯核对，不捏造其缺失的完整编译快照。因此协议显式 cross_binary_replay=true，exact重放是否成立仍需实跑。

完整运行step6已有前面步骤的pipeline cache/allocator历史，单步从新process开始；即便数学源码相同，也不能在实跑之前承诺字节相等。若不同，应先核对这些执行边界，不立刻把剩余差异归为latent/text敏感性。

## 工具与排队

新增 tools/diagnose_native_step_replay.py：默认CPU准备，绑定source result/reference/package/runner及工具；复制六个实际输入；执行时复核frozen prediction sources/输入并完整校验包，命令由固定工厂生成并检查36层/布局/选项。超时会终止整个time/native process group并保留记录。输出是bitwise_equal/status，不引入official passed gate。

准备目录：outputs/q2-chinese-step6-native-replay-v1；执行快照与守护worker：outputs/q2-chinese-step6-native-replay-v1-execution。待root GPU明确放行后运行：

```sh
.venv/bin/python outputs/q2-chinese-step6-native-replay-v1-execution/worker.py
```

沿用递归后代9GiB RSS、整GPU6GiB、host available3GiB、两物理核0/2的guard。此次没有启动worker，不占GPU。4项专用synthetic tests通过：exact/1ULP差异且无official gate，shape/NaN拒绝，bound byte size/SHA/type拒绝，以及固定36层/heads/layout/options命令。未准备2×2网格。


## 已授权单次执行与独立复核

GPU 放行后仅运行上述封存 worker 一次，101.321514 秒完成，退出码 0；已立即通知 root 释放 GPU。结果为 `reproduced_exactly`：actual、封存 expected、历史 fullnative prediction-6 三份数据逐字节相等，max abs = 0，NRMSE = 0，不同 float 数 = 0。共同 SHA 为 `622738cfeb9f811e162ac0c8053ae91917d316a1513856044a8f72539f1f3fe7`。

执行后另用独立 stdlib 检查三份 bytes 相等，重算六输入与 expected 的原始来源和副本 SHA，核对计划的全部 bound 文件、执行快照 5 项及预测器快照 212 项均一致。原生 result SHA 为 `70083c0574432a2ea7c45c85dac2951fa486b60d4bfd1a24eceddb1811e729f0`，plan SHA 为 `78700f98092646f3f86d9b19ec91d22537fabb9d1f00bbeafd0e2a4a3e08a0d3`。完整日志/输入/结果/资源身份另存 task-Q2-native-replay-result.json。执行前工具调用封存 package_model.verify_package 完整验包；执行后没有重复散列大权重。

190 次资源采样：递归后代 RSS 和峰值 867565568 bytes（约 0.808 GiB），整张 GPU 已用峰值 4146 MiB，最小 host available 16612691968 bytes；无守护触发。亲和性限定物理核 0/2，runner 内部仍请求 4 线程，不将其误称 2 线程。RSS 是采样共享页可能重复计数的进程和，不是 PSS 或内核硬上限；GPU 数字为整设备占用，包含既有桌面使用。守护追踪包含另建 session 的原生进程，未仅测 Python wrapper。

该结果关闭本例中“六输入/布局/选项或独立调用不能复现原失败 native 步骤”的疑问；在这一个保存输入上，跨二进制与新进程 cache/allocator 状态没有造成输出字节差异。它不证明所有步骤或其他输入等价，也不归因于某个算子。协议仍 `native_acceptance_eligible=false`，没有 official gate：原中文完整运行 22/25 及 PNG max 13 的失败没有被修复或改写。worker 的 passed 仅指重放命令退出成功。未运行 latent/text 2×2 网格，未修改 runtime、默认模式或门槛。

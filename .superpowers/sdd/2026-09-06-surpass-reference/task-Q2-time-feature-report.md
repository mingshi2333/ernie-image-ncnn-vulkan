# Q2 中文 step6 时间特征单因素诊断准备

CPU 结果：native 与官方时间特征并非完全一致，但差异很小。4096 个 FP32 值中 228 个不同，max `1.4662742614746094e-5`、NRMSE `6.13038486550676e-7`。当前只观察到实现差异；不能将它直接认定为中文轨迹失败原因。已经准备仅替换 in2 的单因素 GPU 实验，本轮尚未执行。

## 精确 native 调用链

完整 native `denoise()` 直接取 `FlowSchedule::turbo(8).timesteps[6]` 并调用 `timestep_features(float)`，中间没有 std::to_string/std::stof。实际生产 schedule 的 raw=0.25、scaled=1、denominator=1.75，最终 timestep 是 FP32 `571.4285888671875`，bits `1141824366`，与官方 teacher-force 保存值一致。

用于额外观察 CLI 边界的小 CPU helper 链接现有 runtime，直接调用上述两个生产函数，不复制它们的数学。它记录 std::to_string 为 `571.428589`，std::stof 恢复相同 bits。再用冻结 denoise runner 对完整 decimal 与 to_string decimal 分别执行 --timestep，三份输出（直接生产调用、完整 decimal CLI、to_string CLI）逐字节一致。因此该 step 的小差异不来自 timestep 浮点参数丢失或字符串往返。

- a4a80b... 是 block-sequence runner，**没有 --timestep**，保持它作为预测执行器。没有假造其 feature 输出。
- feature 生成器是冻结 `ernie-denoise-runner`，SHA `f16a9689615b3023cb83f24404dda1a4f551ae934b22bd31483602904c3e55a2`，与 root 本次完整 runner 配套冻结版本相同。
- 新 CPU observer SHA `b035b29d8482def8225625b6edee80e8cb6e40fac00c135535ca57f6c7ca8f21`。它输出生产 schedule/feature，用于核对直接链路；不是替换 runtime。
- native feature SHA `5919ea0805bf18fa0de6fcb1ebf080cd7e62af72ec411a57f5f9dce71cd604c1`。
- 官方 feature SHA `ae3da3292b1c5813d341b664984b36038537d1c7f3d2cd88eeaa5666774841b8`。
- 最大差异在 index239：native `0.27619609236717224`，official `0.276210755109787`。
- src/denoiser.cpp 与 src/latent_ops.cpp 与先前 a4a80b... 实验执行快照逐字节相同。源、helper、两份 CLI 输出及比较保存在 `outputs/q2-chinese-step6-time-feature-v1`，provenance.json 包含独立 SHA。

源码层面 C++ 使用标量 exp/sin/cos，官方 PyTorch 使用对应 tensor 算子。当前未分离 exp 与 trig 的误差贡献，不能将观察强行归到某一 libm 函数；先观察它对单步预测是否有实际放大才有价值。

## 已准备的严格单因素队列

`tools/diagnose_time_feature_swap.py` 默认只准备文件，不运行模型。准备目录 `outputs/q2-chinese-step6-native-time-swap-v1`，状态 prepared_not_executed。仅 in2 换成 native feature，另外五个输入与 expected prediction 独立重新计算 SHA，确认与已通过的官方同输入基线完全一致。形状、step6、中文 fixture SHA、包 manifest SHA、runner a4a80b...、source snapshot、原有 gates 全绑定。

执行入口再次验证绑定文件、所有 frozen source、input shape/byte size/SHA/finite、与原始命令只允许 fixture/output 替换，并在 native 前重新运行固定包完整校验。采用基线执行快照中的 validate_block_sequence/package_model，固定 precision FP32、Vulkan、stream、36 blocks+heads。不会增加任意 shape 支持或修改 default/gates。

排队命令（需 root 授予 GPU 独占，并沿用递归后代 RSS/GPU/host guard）：

```sh
taskset -c 0,2 .venv/bin/python tools/diagnose_time_feature_swap.py \
  --execute outputs/q2-chinese-step6-native-time-swap-v1/plan.json
```

该工具 --execute 不自带资源硬限；必须包在已审查的递归后代监控 worker 外层，不能把 taskset 当内存限制。现有 native 内部4线程请求不变，以两物理核 affinity 约束 CPU。建议沿用9GiB sampled sum RSS、整GPU6GiB、host available>=3GiB、监测失效终止。此次没有执行该 GPU 命令。

判读：这是时间特征实现差异诊断，不是自由运行 acceptance。替换格相对官方 prediction 的 error 含 time-feature 实现差异与 native DiT 差异，不能冒充“相同六输入”的纯 DiT 数值误差。需与已完成 official-time 基线直接对比，且仍保存全部失败。只有完成后才知道这1e-5级输入差异是否放大。

## 执行快照路径审计

原 diagnose_pipeline_step.py 会复制 ROOT/tools 到 output/scripts，但**普通 live-tools 启动不会因复制动作自动改为执行 output/scripts**。source_snapshot 本身只证明封存字节，不能单独证明所有脚本已导入或冻结了导入时刻；该边界在新报告中明确记录，未抹去旧数据。

本次已完成 teacher-force 的 worker 实际 command 调用 `...-execution/tools/diagnose_pipeline_step.py`；prepare_block 的 ROOT 由该冻结文件路径得出 execution 根，其 ROOT/tools 又指向同一冻结目录。独立 CPU import-path 复现确认 diagnose_pipeline_step、prepare_block、validate_block_sequence 三模块来自 execution/tools，ROOT 为 execution。原实验开始前及结束后212文件都匹配 snapshot.json，因此没有发现“本次执行 live source 却宣称执行冻结源”的实际缺陷，无需改写该历史实验。单因素执行工具也明确从同一个 execution/tools 导入所需辅助模块，且在导入前逐项验证全快照。

## 测试

4项专用 synthetic tests PASS：只有 in2 改变且 gates 保留；修改任一基线 tensor 拒绝且不创建 output；错误feature长度/NaN拒绝；缺输入/path traversal拒绝。CPU helper 已编译并运行、两种 CLI decimal feature 输出已运行比较；未构建共享 build-dev，未运行任何模型/GPU。保持所有原始数据不变。

# 512×384 原生 PE → Vector 文本 → 图像回归

固定历史 PE/apple development fixture、官方 FP32 模块参考、已保存的初始噪声、315 个有效文本 token、8 步 Vulkan FP32 DiT、CPU FP32 direct VAE。生成程序内部实际执行 greedy PE 和全部 25 层文本编码，没有通过 `--embeddings` 绕过文本阶段。

全部 **25/25 张量门槛通过**，PE 完整增强文本和 token IDs 与官方逐字节一致，315-token EOS 正常结束。PNG MAE 为 0.001559787326388889、最大误差 1，原生量化与保存的 decoded 张量精确对应。本次 `native_acceptance_eligible=true`；历史相同完整链路的 decoded 失败（24/25）仍保存在对照轨迹中。

| 边界 | 历史最大误差 | 本次最大误差 | 本次 NRMSE |
|---|---:|---:|---:|
| text | 0.003784179688 | 0.0009155273438 | 3.136188651e-6 |
| final | 0.005950927734 | 0.002828598022 | 9.332130594e-5 |
| decoded | 0.01110547781 | 0.005784600973 | 5.303196947e-5 |

固定 decoded 最大误差门槛仍为 0.01099487681388855。全部 25 个保存的 FP32 边界还与前一份 saved-vector 诊断逐位一致，桥接核验见 `bridge-check.json`；这次的验收资格来自实际原生 PE/text 执行，不借用前一份绕过诊断的资格。

runner SHA-256 为 `ae04f103eaa979f64a09108c9863b39fb81a6ced6dc59a8b5e0db9132ba6aa01`，validator 为 `e5046eb8d1ef00be88c6fe6ca91bde3ad4cd7225a112e34b824b701daa4f67fd`。执行目录封存了 65 个 Python 文件，所有文件在 worker 与结果快照之间的 SHA 相同；这不表示它们全部被导入。`trajectory.json` 经独立保存结果审计工具重新计算，官方输入、模型、源和结果身份均保留。

原生单进程 `/usr/bin/time` 记录峰值 RSS 15448060 KiB、总时长 622.692 秒。此次带完整包散列检查和 trace，主机还有下载及小型 CPU 工作，不能作为受控性能比较。`wrapper-observer.json` 只覆盖 Python 外层进程组，其 771 MB 数字不是整条任务峰值；验证器为原生程序另开进程组，后补的原生观察器记录在 `native-observer.json`，并明确晚于开始加载启动。没有把这些采样当作实际 Vulkan allocation 峰值。

本 runner 在 F1 共享包 pipeline 桥接之前冻结。这证明显式 Vector 模式修复了这一个历史完整原生用例，不证明新的共享包全图已通过，也不证明中文 1024、长文本、低精度或正式 72-case 质量验收。中文 saved-vector 仍为 22/25、PNG 最大误差 13，且其部分误差比已补偿基线更大，因此 Vector 没有成为默认模式。

原始文件：`outputs/pipeline512x384-pe-native-vector-fp32-v1`；worker：`outputs/q2-native-vector-pipeline-worker-v1`。大权重、张量、runner 与 PNG 不进入 Git。

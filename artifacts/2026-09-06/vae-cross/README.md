# PE 图像误差的四组 VAE 交叉解码

本次诊断使用已有 512×384 FP32、完整 PE 增强提示词的真实文生图结果，将官方/原生 VAE 分别作用于官方/原生 unpacked latent。四组均在独立 CPU 进程中完成，未重新生成 DiT latent，未改变历史门槛。

**结论：当前 decoded 超限来自输入 latent 已有的差异，不能归因于独立 VAE 实现。** 用官方 VAE 解码原生 latent，仍在同一像素位置超过完整流水线门槛。四组差分不能解释为线性可加的因果百分比。

| 对照 | 最大绝对差 | NRMSE | 固定门槛结果 |
|---|---:|---:|---|
| 原生 VAE / 官方 VAE，均输入官方 latent | 7.152557e-6 | 2.566869e-7 | 独立 VAE 通过 |
| 原生 VAE / 官方 VAE，均输入原生 latent | 2.920628e-6 | 2.529123e-7 | 独立 VAE 通过 |
| 官方 VAE：原生 latent / 官方 latent | 0.01110604405 | 0.0001002843 | 完整路径 decoded 最大差失败 |
| 原生 VAE(原生 latent) / 官方 VAE(官方 latent) | 0.01110547781 | 0.0001002783 | 完整路径 decoded 最大差失败 |

独立 VAE 门槛保持 NRMSE ≤2e-5、最大差 ≤`.0002 + .0002 * max(abs(reference))`。完整路径保持 NRMSE ≤.003、最大差 ≤`.0002 + .01 * max(abs(reference))`；本例 decoded 最大差限值为 **0.010994876813888551**。两个失败项的最大差位置均为 NCHW `[0,1,88,302]`，交互项最大值为 5.960464e-6。

本次 `official_decoder(official_latent)` 与历史官方 decoded **逐位相同**；`native_decoder(native_latent)` 与历史原生 decoded 也 **逐位相同**。因此结果并非不同版本、卷积策略或重复运行改变造成。原生固定为 CPU FP32 direct convolution 和当前 GroupNorm，所有输入、模型包、runner 与相关源码散列记录在 [results.json](results.json)。诊断脚本快照保存在 [diagnose_vae_cross.py.snapshot](diagnose_vae_cross.py.snapshot)。

复现入口（在项目根目录或已连接相同模型/输出数据的隔离工作树）：

```sh
.venv/bin/python tools/diagnose_vae_cross.py \
  --run outputs/pipeline512x384-pe-fp32-v1 \
  --model models/turbo512x384-s2048-portable \
  --output outputs/vae-cross-pe-new
```

四次运行原始日志、复制的输入和输出位于本地 `outputs/vae-cross-pe-v1/`，大张量不进入 Git。测试验证四路形状/有限值、输入与解码误差分离、交互项，以及平均误差不能掩盖局部最大误差失败。

下一步用保存的官方文本特征运行相同去噪轨迹，判断原生文本条件的误差贡献，然后继续隔离 DiT 的首次偏离。该实验属于诊断；历史完整图像仍为 24/25 张量通过，不宣布质量问题关闭。

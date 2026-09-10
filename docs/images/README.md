# 演示图来源

这三张图片直接复制自本项目已经完成的原生生成结果。原始 PNG 未裁剪、调色、缩放或重新编码；README 中的显示尺寸由页面控制。它们分别来自对应报告的历史版本，并非为改写 README 重新生成。

三例均为 Turbo、8 步、Vulkan FP32 DiT、原生 CPU 文本编码和 CPU direct VAE，PE 关闭、CFG=1。它们使用对照实验保存的初始 FP32 噪声；只设置相同整数 seed 无法保证复现这些像素。每例的实际命令、模型和程序来源可从下方报告追溯。

| 图片 | 尺寸 | 张量检查 | PNG 平均差 / 最大差（0..255） | 实验记录 |
|---|---|---:|---:|---|
| [apple-1376x768.png](apple-1376x768.png) | 1376×768 | 25/25 | 0.000617291 / 1 | [原始记录](../../artifacts/2026-09-07/runtime-large/README.md) |
| [cat-1024.png](cat-1024.png) | 1024×1024 | 24/25 | 0.002676964 / 1 | [原始记录](../../artifacts/2026-09-06/attention-parity/runs/pipeline1024-long-s64-fp32-chunked-v1/result.json) |
| [lake-1024.png](lake-1024.png) | 1024×1024 | 21/25 | 0.022625605 / 13 | [原始记录](../../artifacts/2026-09-06/attention-parity/runs/pipeline1024-chinese-s64-fp32-chunked-v1/result.json) |

苹果的数值检查全部通过。白猫的最后一步预测未通过最大误差检查，图片检查通过；湖泊有晚期预测和解码未通过项，图片最大差 13 也超过原上限 2。展示图片保留这些未通过结果。

## 原始提示词

**apple-1376x768.png**

> A red apple on a wooden table, soft daylight, realistic photo.

**cat-1024.png**

> A small white cat sitting beside a blue ceramic teapot on a wooden desk, warm afternoon sunlight through a window, a green plant in the background, soft shadows, realistic photograph with fine detail.

**lake-1024.png**

> 雪山脚下的蓝色湖泊，松树林，清晨阳光，写实风景摄影。

文件摘要与原始路径保存在 [provenance.json](provenance.json)。模型、程序和其他生成产物继续留在本地；这三张经用户要求选入 README 的图作为文档资源提交。

# 固定执行快照后的 VAE 交叉复验

本轮为 Q1 工具审查修复后的真实复验，数据位于 `outputs/vae-cross-pe-v2/`。四个 CPU FP32 解码进程均正常退出；运行脚本及全部 56 个本地 Python 依赖先复制，再从快照执行，执行前后核对快照散列。模型资产通过显式根目录读取，仍按固定来源清单校验。

本轮与 [首轮交叉结果](../vae-cross/README.md) 数值完全一致。官方/原生历史 decoded 分别与 `oo`/`nn` 按 dtype、shape、实际 bytes 比较，均逐位相同。新增正负零测试防止用浮点数值相等冒充 bytes 相等。

| 比较 | 最大绝对误差 | 结论 |
|---|---:|---|
| native decoder / official decoder，相同 official latent | 0.000007152557373046875 | 独立 VAE 门槛通过 |
| native decoder / official decoder，相同 native latent | 0.0000029206275939941406 | 独立 VAE 门槛通过 |
| official decoder，native / official latent | 0.011106044054031372 | 超过连接门槛 0.010994876813888551 |
| native/native 与 official/official | 0.011105477809906006 | 连接门槛仍失败 |

结果支持沿输入 latent 继续定位。它不代表完整原生 PE 图像质量项已关闭，不修改既有门槛，也不以交互项计算线性因果百分比。原始首轮快照与报告保持不变。

验证：6 项交叉报告行为测试通过；四组模型复验成功；两个历史结果逐位复现。`results.json` 保存全部 runner、worker、来源、输入输出散列和独立进程命令。后续工具变更不能改写本记录。

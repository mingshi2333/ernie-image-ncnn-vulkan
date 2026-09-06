# Q2 step6 单因素条件敏感性准备

状态：仅 CPU 准备，两个独立 worker 均未启动，无 GPU 使用。F2 当前独占 GPU。前置 native replay 在提交 0aeea71 已逐字节重现历史中文 prediction-6；本次准备不会改变或覆盖该证据。

## 可区分的问题与解释边界

A 固定实际 native 文本、time、RoPE 和 mask，只把输入 latent 换为官方 step-5。这测量给定 native 条件下，将已有累计 latent 状态差异消除的局部效果；该状态差异可能包含过去文本、DiT、scheduler 等全部历史因素，不能称作纯 DiT 误差。

B 固定实际 native step-5 latent、time、RoPE 和 mask，只换官方 padded-text。这测量在失败 native 状态下当前这一步的文本条件敏感性；不会撤销文本在前六步已经造成的 latent 变化，不能称作全部累计文本贡献。

两项分别对原 native prediction 和原官方 prediction 描述 max abs/NRMSE/不同 float 数/bytes 相等性，没有 official quality gate、没有 passed 门槛。混合输入不是官方六输入组，因此相对于官方 prediction 的距离不能标作 pure rounding error。两项条件效应不可直接相加而声称非线性交互已完全拆分；本轮没有准备两者同时替换的网格格点。

## 固定输入身份

所有张量均连续 little-endian FP32；去掉官方 metadata 的 leading batch=1 只改变逻辑描述，无字节转换。

| 输入 | 原 native SHA-256 / 固定值 | 替换值 |
|---|---|---|
| in0 CHW [128,64,64] | 4fb0226e69ac8cd3e743ac39faa2a46c19b60d23adaefcf536371e875cb388f9 | A: 8368a393b89407618726df770f3015892626762e90a986bc851d32d7a85d48f4 |
| in1 [64,3072] | 2a515b8bb4c25d95851e4bff37e6a7f36464daa0c7017918cbcec43c22d97dd3 | B: 40c6b63478d16f90370925bfac16381661bef12a48ae197b59bdc5ae1a1f316d |
| in2 [4096] | 5919ea0805bf18fa0de6fcb1ebf080cd7e62af72ec411a57f5f9dce71cd604c1 | 两项不变；timestep 571.4285888671875 |
| in3 [4160,128] | 18ef5d09d02a0c9a7740183406d6ef2cbd85dfc7cd5d59e625200a831086f49c | 两项不变 |
| in4 [4160,128] | b392037350ef3f0639f940a20f30dbbea129ed33fb14e22ae81a0cb1f9890e69 | 两项不变 |
| in5 [4160,4160] | 12f5c9c117a285813a3677969cd181090513e8eae36ac787c7749fdb5c5319cc | 两项不变 |

原 native prediction-6: `622738cfeb9f811e162ac0c8053ae91917d316a1513856044a8f72539f1f3fe7`。
官方 prediction-6: `4dac3a98973693a7c94438f8f9760411eabedd239736e85744d73b922a5dc127`。
官方 fixture: `81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7`。
预测 runner: `a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc`。
Package manifest: `72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1`。

## 封存与执行边界

新增 tools/diagnose_native_step_sensitivity.py，协议 native-chinese-step6-single-factor-v1。强绑定已成功重放的整份 baseline plan/result SHA、已审查官方 fixture/替换张量 SHA；验证精确六输入及只有所选一个 SHA 改变，shape/dtype 不变。执行前重新验证输入副本和原封存预测器 source/runner、完整模型包；固定 36 blocks、两 heads、stream Vulkan FP32、64×64 packed shape、64 text tokens。执行时只用封存 helper 和封存 package verifier，不导入活动 tools。新 worker 在启动诊断前重验自身、两个 helper、plan 的 SHA。冻结快照状态保留 prepared，不后改；执行另写结果。

| 实验 | plan SHA | execution-snapshot SHA |
|---|---|---|
| official-latent-only | 5bfecc95246287591034edf6c1be54f2b97da27b531027c0270f2501ce1e96ca | 40ca4f4c24b45e3013d9aba0def719b8fe5111a2c3380ff78595c6a43a97fa52 |
| official-text-only | 44f662b2f5274e37436edcf0b0a55861bfff094c0e8c0a28643cf8e7a390af76 | 4b158d854726727e7d4e76a486eedfa5bd7271d5c93a3286f820632f9c842d98 |

每个 worker 均为独立一次执行；只有在 root 明确 GPU 放行及选择实验后才运行，不可同时运行：

```sh
.venv/bin/python outputs/q2-chinese-step6-official-latent-only-v1-execution/worker.py
.venv/bin/python outputs/q2-chinese-step6-official-text-only-v1-execution/worker.py
```

资源守护仍是两物理核 0/2（runner 内部请求 4 线程）、递归后代 RSS 9 GiB、整 GPU 6 GiB、host available 3 GiB。监视器超时也作为不可用处理并停止子进程；native 超时终止其独立 process group。worker success 只表示执行与结果生成成功，不代表质量通过。

4 项专用小测试通过：两个允许单因素、拒绝第二处变化/缺输入、拒绝错误替换/shape/dtype、输出描述包含两种距离且不含 passed gate。准备实际副本时校验所有输入/两份预测的 size/SHA/finite。没有运行模型、没有修改 runtime/default/gates。

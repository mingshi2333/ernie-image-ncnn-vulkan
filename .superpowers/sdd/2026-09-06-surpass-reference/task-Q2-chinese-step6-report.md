# 中文第 6 步官方同输入 teacher-force 实验

单次实验 PASS。独立从 actual/expected 原始 FP32 bytes 以 FP64 复算，NRMSE `1.2358507024489089e-5`，最大绝对误差 `0.0004429817199707031`，error L2 `0.010200039352897849`。固定门槛未改：NRMSE <= 0.003，max <= 0.0002 + 0.01 × reference_max = `0.054409232330322264`。这是一个同输入预测的局部诊断；native_acceptance_eligible=false，不替代中文完整 8 步轨迹/PNG gates。

## 身份与实际执行

- 运行输出：`outputs/q2-chinese-step6-current-teacher-v1`。
- 实际执行目录：`outputs/q2-chinese-step6-current-teacher-v1-execution`。先复制当前 source_inventory 的完整文件与 runner，实际调用这里的 `tools/diagnose_pipeline_step.py`，没有执行活动 tools 目录再声称封存文件就是已导入来源。共 212 项源/runner 身份逐项在结束后重新哈希确认匹配；这是封存清单，不代表全部文件都被导入。
- runner SHA256 `a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc`。root 完成当前 F1 组件重构后的构建，此次保留该二进制；并非旧 saved-vector runner。
- 中文 reference fixture SHA `81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7`。
- package manifest SHA `72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1`，`models/turbo1024-s64-portable`，1024×1024/text bucket64/36 DiT blocks。
- 零基 step6，timestep `571.4285888671875`。输入 latent 是官方 step5 输出；文本、RoPE/mask 同官方 fixture，time features 由既有官方 diffusers API 在 CPU 生成。图像头、36块、输出头均 native Vulkan FP32/stream。
- 六份输入 byte size/SHA 校验，expected prediction SHA 校验；独立复算检查 128×64×64 FP32 shape、有限值、NRMSE/max 与工具 JSON 精确一致。
- actual SHA `1a1c133a28a64362c5efb83816e5d0a390d205fc517ea86a4f4e6a7adb8d57a9`；native result SHA `66f6009ecbdda647174ca5c1579a358e93906a28c8514859c7ee5806ba43b337`。配套 `task-Q2-chinese-step6-result.json` 绑定 fixture、logs、worker、snapshot 和结果哈希。

## 与既有自由运行证据的关系

| 条件 | step6 NRMSE | step6 max | 固定门槛 |
|---|---:|---:|---|
| 本次当前 runner，官方 latent/text/time | 0.00001235851 | 0.00044298 | PASS |
| 既有 chunked 中文自由运行 | 0.001371368 | 0.08202004 | FAIL max |
| 既有 saved-vector 中文自由运行 | 0.002599021 | 0.09975964 | FAIL max |

本次局部误差远小于自由运行第 6 步偏差，反对“该步在精确官方输入上仍有足以独立越门槛的 DiT 误差”这一具体假设。它支持进一步研究输入轨迹/文本扰动如何放大，但不提供因果比例：runner 不同、teacher-force 一次性归零了所有先前 latent 偏差且使用官方 time/text；局部小误差仍可在后续积累。没有证据据此断言 Gemm、attention、残差或 timestep 哪个支配，也不能说文本误差已完全解释结果。

最有区分度的后续仍是同一个 frozen runner、同官方 time/rope/mask 的 step6 2×2 输入实验：latent=official/vector-step5，text=official/vector；先控制其中一个轴再测交互。只有 official/official 格具有当前 exact oracle，其余格是敏感性反事实，不可把原官方 prediction 冒充这些输入的正确 oracle。需要各格单独 provenance。此次只执行了一个授权实验，没有擅自继续四格或修改数学/shader。

## 资源与失败边界

启动前 host available 约17.6 GiB、整GPU约1692 MiB。以进程 affinity `{0,2}` 限制两个物理核；现有工具和 runner 内部仍请求4线程，明确不是“改成2线程”。进程继承该 affinity。环境 OMP/BLAS/MKL 设2，未修改 runtime。

外层 worker 遍历全部递归子进程并缓存 PID，因此覆盖 run() 另起 session 的 `/usr/bin/time + native`，不是只量 Python。约0.5s间隔采集进程树 RSS 总和、host available、整GPU；9 GiB RSS/6 GiB整卡/host available3 GiB/测量失效/2400s超时均触发终止并保留记录。RSS 是相加值，可能重复计算共享页；采样有间隙，不是硬 cgroup 上限或精确 allocator 峰值。

实际 worker 101.09s，return0，无 stop_reason：进程树采样RSS峰 `1,674,084,352` bytes（1.56 GiB），独立 GNU time native maxRSS `1,220,268` KiB；整GPU峰4068 MiB，内部100ms原始GPU采样另保留 native 日志。无超预算/模型崩溃/重跑；GPU任务结束后回落1681 MiB并通知 root 释放。该时长包含校验/导入/准备/运行/比较，不是无trace正式性能基准。

历史完整中文 saved-vector 22/25 + PNG max13 的失败保持不变；本次不修改旧 artifact 或 gate。

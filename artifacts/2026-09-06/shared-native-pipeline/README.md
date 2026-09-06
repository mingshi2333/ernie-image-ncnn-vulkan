# 共享模型包的完整原生生成验证

512×384 共享模型实例已真实执行完整 greedy PE、25 层原生 Vector 文本编码、8 步 Vulkan FP32 DiT、CPU FP32 direct VAE 和 PNG 写出。独立复核结果为 **25/25 张量通过、PNG 通过**。全部 25 个 FP32 边界及 PNG 与前一份固定模型包完整原生运行逐位一致。

本次直接使用 schema3 `objects/<SHA256>` 文件，由 C++ `ModelPackage` 选择实例并向组件传递内存 param 与 CAS 权重路径。运行时没有 Python、临时 param 或按分辨率重新复制权重。原生可执行文件 SHA 为 `6a3481e558a003e3f6fad6d8a9bc2ff903216d539c7ce2b86b352c19befaecbc`。它包含 F1 组件连接，在 F2 生产图生图接入前冻结。

| 固定身份 | 值 |
|---|---|
| 共享包 manifest | `dcf566507557ac1056cf4a8d2620c697920f49a0d391dfeaf6e692c5678424a0` |
| 选中的原 schema2 源 manifest | `ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59` |
| 官方完整参考 fixture | `8d7638b1808b4b435565d4045ffc39e3a06fb1de35f8a52f23de75aec2feae11` |
| PE 模型 manifest | `bf6b5a8314ba5abe3654c95eabfb8f4e4a8a8541fff6ebea3c95a9f7683a8aff` |
| 初始噪声 | `954333d9dad2fca3a09e5ad630e1aceb5a951a592c2789489f64a40bf0e3a146` |
| 最终 PNG | `24353ce4881e0f6f3a5b116d93fa7d898cbda885847a2b6b7e80da411eb90f85` |

原始提示词为 “A red apple on a wooden table.”。PE 实际输出 315 tokens 并遇到 EOS；完整增强文本与 IDs 和官方参考一致。随后运行原生文本编码，未使用 embeddings 旁路。PNG 与官方参考的 MAE 为 `0.001559787326388889`、最大误差为 `1`，原生 decoded 到 PNG 的量化检查精确通过。所有数值门槛保持原值。

对照为 [固定包完整原生结果](../native-vector-pipeline/README.md)，不是 saved-embeddings 诊断。`bridge.json` 逐项重新计算两份运行的实际张量 SHA，并与各自记录核对，才判定逐位相同。

## 执行与独立审计分开记录

`native-source-snapshot.json` 记录 runner 与 213 个封存源文件。`execution-snapshot.json` 记录实际执行的验证工具和 215 个文件；它在 native 冻结源之上只新增验证适配器/测试并更新 validator。文件归档不表示每个文件均为实际编译或导入依赖。

独立审查发现原执行验证器缺少可信参考来源绑定和完整步骤分母检查。冻结执行没有被替换；当前实际输入原本就有完整 25 项，并属于此前经过审计的官方 fixture。修正代码在 `a77dccc`，由独立 reviewer 关闭三个 Important 后另行执行 `independent-audit.json`：

- 重新认证共享包的全部 85 个 CAS 对象，保持每个实例的 136 项原始 runtime 绑定；
- 用受版本控制的整份 fixture SHA 关联固定源 manifest，重新认证源 manifest 对象及完整文件表；
- 强制完整 conditioning、每一步 prediction/Euler 和 final/unpacked/decoded 共 25 项，拒绝缺失、重复、suffix、错误类型/形状；
- 重新核对 runner、执行脚本、官方输入、PE 文本/IDs 和图像身份，并独立重算全部误差。

`auditor-source-snapshot.json` 是此次独立审计所用源，不能当作原生成进程的执行源。当前共享验证的 oracle registry 只收录已审查的 PE/apple development 参考；新增参考要先建立官方执行和来源证据，不能仅填写自洽字段。

## 资源与适用范围

原生 `/usr/bin/time` 记录峰值 RSS `13749060 KiB`、原生总时长 `721.751 s`；外层从启动即跟踪递归后代，采样 RSS 总峰 `14389813248 bytes`、整GPU峰 `3510 MiB`，资源 guard 未触发。后者包括验证器、time wrapper 和 native，不能称为单进程 RSS 或实际 Vulkan allocation 峰。

本轮有 trace、两层完整包验证、CPU affinity 和并行受限构建/审查，主机也存在 swap 活动。它不属于受控性能轮，不能用与历史运行不同的时长或 RSS 推出速度/内存优势。

共享测试包包含原 512×384/s2048、1024×1024/s64 两个实例；本报告只实际生成前者。测试存储用硬链接复用本地不可变源对象；生产 builder 按对象复制一次，这里没有把硬链接的磁盘占用当作产品节省测量。任意动态尺寸、1024 共享实例实际出图、图生图、正式 72-case、人工盲评和跨平台完整模型验收均不由此报告关闭。Vector 仍为显式候选；中文及其他历史失败保持记录。

原始输出：`outputs/pipeline512x384-shared-pe-native-vector-fp32-v1`。执行目录：`outputs/f1-shared-native-worker-v1`。独立审计：`outputs/f1-shared-native-audit-v1`。大型张量、图片、模型和二进制留在本地输出目录，不进入 Git。

`manifest.json` 固定本目录八份 JSON 证据的散列和大小，便于复核报告副本；原始运行身份仍以各自执行快照为准。

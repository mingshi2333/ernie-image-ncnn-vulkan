# 固定对照项目的权重身份与差异

对照源为 `futz12/ernie-image-ncnn-vulkan` 的 `8dcd6e4411137d8abe92c9d78581c4c96d5182c6`，其 ncnn 为 `f6f734f44d66f469fefee9ee401fd1cb5e3d573e`。28 份完整模型资产来自固定 Hugging Face revision `140a052f7919f279de7f697fa54f33bd1c0cac2b`。官方组件为 Turbo `bc68c81e2a1730a394d5fc9fae70713dee940140`。

目前可以证明大量实际权重内容与明确的层角色，但仍不能声称整个模型图完全等价，也没有取得对照项目完整成图或配对速度结果。

## 实际完整扫描

独立 CPU 进程读取全部 102 个经过来源验证的官方组件，得到 1,123 个官方张量；解析对照 10 个完整二进制权重文件直到 EOF，共 1,129 项序列化记录，零解析缺口。28 份资产还包含图、tokenizer 与元数据，并非 28 个权重文件。

| 结果 | 数量与含义 |
|---|---|
| 官方 FP32 规范化内容直接匹配 | 1,121 项 |
| 其中唯一内容匹配 | 1,101 项 |
| 内容本身有多个相同候选 | 20 项；包含 PE/text 部分相同 RMSNorm 与 tied embedding/LM head |
| 无整张量直接匹配 | 8 项；后续单独验证拆分投影与派生常量 |

扫描耗时 218.78 秒，GNU time 最大 RSS 52,612 KiB；3 GiB / swap0 / 两核额度内完成，exit 0。分块规范化缓冲上限为 4 MiB。这里统计的是离线权重扫描，不是模型加载或推理内存。

`scan-execution.json` 保留原始命令、102 个组件名和两份实际工具 SHA。原始完整结果的 SHA 是 `75fb0e4dc069f8cb1da2cf2cd052d7f405eb6d1a73fdc5b91808ce1fc46644c4`。当时的扫描器没有后续 graph-role 扩展；原输出和执行源保持不变。

## 独立图角色复核

新的 `port_graph_contract.py` 按固定图的块顺序和实际连接指定角色，再核对已扫描权重值和官方逻辑张量，避免从相同散列反推角色。

- DiT 36 层、文本编码器 25 层、PE 26 层及最终 norm、三个 embedding/LM head 入口：**859/859 项 named weight 对应通过**。
- 检查 Q/K/V 到 attention 的输入槽、Q/K norm、attention 输出投影、MLP gate/up/down、两个 residual、DiT 条件调制与 PE 每层 cache 输入输出。
- 检查矩阵输出/输入轴、转置和 bias 表达、norm affine，以及逐层权重分母。前述 20 项内容歧义均由图角色和命名组件区分。
- 文本图声明 958 个 blob 槽而实际产生 956 个，保留这两个未使用槽；所有实际边必须指向此前已定义的 producer。其余五图没有未使用槽。

本次补充复核重新读取六个实际固定 param，并使用整个 SHA 认证的先前 full-scan / official-inventory 副本，没有再次读全量大型权重。`graph-execution.json` 绑定这条证据链；`graph-roles.json` 列出全部对应及每份图的 SHA。小型测试覆盖换线、错误轴、错误命名权重/形状、缺少层和重复条件常量。

这 859 项之外，既有 VAE 审计另有 16 项明确的 attention projection 对应。VAE 其他卷积等角色与剩余 heads 的完整逻辑证明仍需扩展；不得把以上数字改写为 1,129 项完整图等价。当前补充工具也不证明 ncnn kernel 的舍入方式、动态 conditioning、全流程数值或速度门槛。

## 八项非直接匹配与明确差异

独立关系工具不信任旧扫描器自报的 offset，重新认证并解析四个相关权重文件、270 项记录和官方组件，结果如下：

| 对照项目中的记录 | 已证明的关系 |
|---|---|
| finalizer 两个 Gemm 的 B/C，共四项 | 与官方 `final_norm.linear.weight/bias` 前后 4096 行逐字节一致；固定连接证明前半是 scale、后半是 shift |
| encoder BN slope / bias | 分别为 128 个 FP32 1 / 0，逐字节一致 |
| decoder 的逆 BN depthwise scale | 逐字节等于 `sqrt(running_var + 1e-4)` |
| preprocessor 的 2048 项 timestep 频率 | 对当前固定官方 CPU 实现有 88 项、各 1 ULP 的差异，最大绝对差 `5.960464477539063e-8`，NRMSE `2.0191671970545926e-8` |

因此八项中七项有 exact 关系，频率一项不能标 exact。频率值为何产生这 88 处差异，没有原导出环境执行证据，不能确定历史原因。

**decoder 的数学常量还有实质区别**：对照使用 `1e-4`，固定官方 ERNIE pipeline 的 inverse BN 明确使用 `1e-5`，本项目跟随后者。上表证明的是对照实际存储值，不能将其写成官方解码公式完全一致。编码端配置 `1e-4` 与解码端公式 `1e-5` 分别记录。这些差异的最终图像影响仍须真实配对运行衡量。

详见 `relations.json`、`frequency-differences.json` 及工作报告 `task-B2-relations-report.md`。

## 原始对照程序的容量边界

固定源码的预检显示其 low-vram 为 host-weight 分配偏好，不是逐块流式加载；文本编码器、DiT 与 VAE 的权重生命周期重叠。即使乐观地将持有的全部权重按两字节估计，也需 22,791,435,590 bytes（21.23 GiB），高于本轮 20 GiB RSS 预算；FP32 为 45,582,871,180 bytes（42.45 GiB），尚未计激活和 allocator 开销。原 CLI 未提供可启用的 FP16 配置。

`capacity-preflight.json` 保存源码、图、runner 与计划参数。这是源码和存储量预检，**不是实际 RSS、OOM 或速度观测**；未以故意触发 OOM 来补一条失败，也不能把无法测出的速度当作无穷慢。当前 `allowed_to_close_S=false`，共同质量和实际配对 S/M 均未关闭。

所有大型原始文件保留在 `outputs/reference-port-v1/`。本目录仅保存 JSON 证据副本，`manifest.json` 固定其大小与 SHA；没有模型、图像或二进制进入 Git。

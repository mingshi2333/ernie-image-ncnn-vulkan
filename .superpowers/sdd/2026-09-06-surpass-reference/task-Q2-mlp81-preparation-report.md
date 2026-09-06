# Q2：81 / matched-input MLP 单一实验 CPU 准备

状态：CPU 全张量分析、实际权重/图认证、observer 独立编译、封存执行合同完成；**未运行任何新 GPU forward**。本实验不改变生产数学、默认、门槛，也不是中文 free-running 验收。等待 Root 独立合同审查及 GPU 排队。

## 为什么选择这个边界

已认证官方 75/87/88 的前一实验（b083fc6，独立审查 8f5f878）证明：改进 down 的 FP64 数学准确度并未改善完整官方 block parity。六行的 native/official down 自身 FP32 误差高度相关（cos≈0.83735），候选消除此相关误差后完整 out0 更远。这是有效反证，不继续调 split-K。

本轮只复用既有 native 堆叠输入 S 与 exact-official 输入 T 的完整 CPU tensors。`outputs/q2-upstream-branch-cpu-v2/result.json` 定义 S/T 为**同一 native block**的不同实际输入，不将其差异冒充同输入实现舍入误差。状态分母 17,039,360，87 分母 51,118,080；两套残差重建均逐位成立。

| 完整向量差异 | L2 |
|---|---:|
| 传入 block14 hidden，S−O | 0.5377929787 |
| 第一残差 S75−T75 | 0.5688275845 |
| attention 分支响应 `(S75−T75)−(S14−O14)` | 0.3074459089 |
| MLP 分支响应 `(Sout−Tout)−(S75−T75)` | 0.5423054235 |
| 最终 Sout−Tout | 0.6766809837 |
| FP32 gate×down 分支响应 | 0.5421905476 |
| 第二次残差加法舍入响应 | 0.01130325746 |
| 87 输入差异 | 0.02922311392 |

input/attention cosine −0.1819845，first-residual/MLP −0.2589531，attention/MLP −0.2515981。非正交，不能换算独立误差贡献百分比。MLP 在这一既有方向的响应更大，因此选其入口81切分继承误差与局部实现差异；**不声称 block15 是堆叠误差的最初来源**，block14 已不同。

CPU 分析 v2 用时1.32s，RSS最大1,035,152KiB，进程swap0，CPU0/2与BLAS2。v1缺少S75既有身份条目而拒绝，原脚本/失败记录保留，未产生有效数值结论。

## 81 的精确图和官方映射

固定 param SHA `e8ed770153823bc60b2f27cfac14b9a58f534eb6bd9f19dc692d650f7faddd13`、bin SHA `0321f883d79cab6606ac3d2736344043625f3a0fed9bb1f5fcfdd33d1acd07da`、官方组件 SHA `d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2` 均实际重算。

完整图链为：75 → Split → `rmsn_11` learned RMSNorm(width4096,eps1e-6) →78；`in5+1`→79；78×79→80；80+in4→**81**。因此81已经过 scale_mlp/shift_mlp conditioning，**绝不把原始75当作MLP输入**。

81→Split为82/83；`gemm_4(83)`→84，对应官方 `mlp.up_proj`；`gemm_5(82)`→85，对应 `mlp.gate_proj`；`ErnieGELU(85)`→86；84×86→87；`gemm_6(87)`→88，对应 `mlp.linear_fc2`。in6只用于随后 `75+in6*88`，不属于 MLP 模块输入。

完整 bin 436,241,436 bytes 恰好消费完毕；不是按矩阵大小猜方向。根据实际 transB=1 与固定 loader 布局，认证到以下官方具名张量，canonical 为完整 little-endian FP32 行序：

| Native | 实际 byte offset | 官方名后缀 | 官方 shape | 完整 canonical SHA256 |
|---|---:|---|---|---|
| rmsn_11 gamma |134235152|adaLN_mlp_ln.weight|4096|a73506e009f4aa1172c78d2eb57eef173cc3a1d8865381532328133af016f285|
| gemm_4 B |134251540|mlp.up_proj.weight|12288×4096|0039d65f89e8f12d679f9d9ead7ad927ab64a7b21be8a3c1b9e0f3ff77d73fbb|
| gemm_5 B |234914840|mlp.gate_proj.weight|12288×4096|b5b8bfd5ff842cf432fb4fa5a26e154533fd26ac6e98c7916619c7272f8d08b8|
| gemm_6 B |335578140|mlp.linear_fc2.weight|4096×12288|58e616f4f101fcd829a165c77196f07a6e13d9e1fd857ecd48d263b16bde2c69|

三矩阵各50,331,648值，BF16仅为磁盘存储，按固定 modelbin.cpp 的 `0x01348B83` 展开认证；norm4096值F32。v1准备器没识别该tag而failclosed，保留failed-prepare.py/failure.txt；v2按实际loader补充后通过。此权重认证为 bounded streaming，未加载模型、未实测该短准备进程的峰值，不能借 CPU 分析的RSS冒充它。

磁盘统一BSH=[1,4160,4096]，native Mat WHDC=[4096,4160,1,1]，官方实际模块输入SBH=[4160,1,4096]；81与75/out0各17,039,360元素。87为BSH=[1,4160,12288]，51,118,080元素。完整字节匹配后才解释边界。

## 不变性与最多三次调用

1. 新native observer只给旧observer的allowlist追加81。五个数学实现、其头文件和libncnn均与旧封存版本一致；独立编译不修改生产CMake。完整out0必须逐位复现 `175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3`，同时严格核WHDC和字节数。失败即停止，不进入official。
2. 官方仍调用原 `block(x,angles,temb,attention_mask)` 一次；额外在 `block.mlp` prehook抓81，原75/87/88 hooks不变。必须四个旧完整SHA均复现，才接受81：out0=`a3f77e…`、75=`091892…`、87=`384328…`、88=`3a1eb9…`（完整值在合同与代码）。baseline文件先独立落盘，结尾再次重验，后续调用不覆盖它们。
3. 若81已逐位相同，**省掉尾支路**，旧official87就是同输入参考。若不同，最多再调用一次原 `block.mlp(native81.transpose(0,1))`，保留相同模块实例、权重、FP32 dtype、backend与TF32设置；捕获其完整87（同时保留88结果身份）。不跑attention/conditioning第二次、不用倒推target，也不修改torch数学。

将 `M87(N81)−O87` 记为81输入差异在官方MLP中的传播；将 `N87−M87(N81)` 记为**真正同输入的81→87实现差异**。两者相加重建原N87−O87。若只发现81不同，仍不能归因RMSNorm，因为75已不同。若matched局部差异小于传入响应，则下一步应转向75→81继承/调制；若局部差异明显，则后续仅选择up/gate/GELU中最有区分度的一项内部边界。当前不预判结果，不按看完结果重设接受门槛。

## 冻结与执行入口

`outputs/q2-block15-mlp81-v2`：

- plan SHA `c7447c7150838dba7d98adf1a8dbca572b857511a341b505d4ac7e35077e1b02`
- worker SHA `2ac4e0c9610b2521784fea8b3ab7c664fcf91e6830e16434b4639de1a557b931`
- runner SHA `3689bfc291423a691e88b43041ef8ad3c6b8e171a907a4758202f78f98bc99c9`
- 5885个bound文件已完整CPU前置复验，含旧输入/angles/constants、官方runtime记录中的实际文件、全model bytes、新source/runner、实际编译依赖。

原官方 make_inputs 的角度/cos/sin/mask准备证明直接复用已审查v3，绑定原plan与实际全部输入字节；没有重生成新的随机输入。项目Python从新封存execution/imports执行；外部torch/diffusers仍从已认证安装路径执行，前后另存实际import/mapped库快照，不谎称从runtime对象归档导入。config必须逐字段等于旧actual torch-config.json。

准备时把audit导入移入CPU proof函数，避免正式执行意外导入live工具；final plan绑定新execution副本，contract.json保留当时CPU证明身份，未回写历史SHA。封存计划构建脚本另存execution/freeze-plan.py，其SHA在evidence中。

审批后唯一命令（**本轮未执行**）：

```sh
.venv/bin/python outputs/q2-block15-mlp81-v2/execution/worker.py
```

worker前置核plan/sequenceSHA；sequence只允许新started.json，失败不重试。native与official进程串行。沿用递归后代RSS9GiB、整卡6144MiB、host floor3GiB、过程swap0、2400s、0.5s采样与CPU0/2守卫。最终状态单列native_full_block=1、official_full_block=1、official_mlp_only=0或1；不把额外MLP调用隐藏成“一次forward”。

## 检查与范围

独立observer编译exit0；三项合同测试覆盖四旧边界任一缺失/错SHA、native完整out0失败、81相同省略tail/不同才需要tail；两项CPU分析测试覆盖分母不完整及非正交响应解释，全部5/5通过。真实新增81数值和matched MLP **pending**。严格official parity与更高数学精度可非单调，前例不证明门槛不可达；不放宽任何门槛，不提升候选，不宣称中文失败修复。

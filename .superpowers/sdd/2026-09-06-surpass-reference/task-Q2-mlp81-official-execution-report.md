# Q2 81 matched-input：完整执行与全张量误差分解

**唯一v5尝试成功完成，GPU已释放。** Root d51b29d独立准备审查通过后，执行者外部核验launcher完整SHA，再执行封存入口。session94611；专属scope `ernie-q2-mlp81-official-v5.scope`，37.3240411s，exit0，无guard停止。实际新调用数：native **0**、官方完整baseline **1**、官方matched MLP **1**。没有自动重试、扩大网格或改生产数学/默认/门槛。

原0c9247d的v2 swap停止仍为负结果。此次复用该尝试中已独立认证的native81/out0，没有把整个旧尝试重标成功，也没有重跑native。

## 前置不变性和执行身份

官方baseline完整out0/75/87/88逐位复现全部旧SHA，才接受新增81并允许额外MLP：

| 边界 | 完整SHA256 |
|---|---|
| out0 |a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197|
| 75 |09189225db6307daa124c88dce40dfc257e8af49d9cea125b44c6c64072b69af|
| 81 |744be265796164ce51d15ac22513870bee07c794f30a5c9089b0f6af2c90f1a6|
| 87 |384328e7033248ef5c43d1a9494d4f93c7e13556bd0913bdda786cfc1f1f10dd|
| 88 |3a1eb976b4772659e9461745bb21ec9fa92582adfd98d90f728aab83d8375ab8|

复用native81为`da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54`，完整out0仍`175bfa…`。81不同，因此额外一次原官方MLP使用**完整post-conditioning native81**，权重、FP32、backend和输入布局转换保持合同。其87 SHA `9a34cfce2296c4f2b3ef7f48aad848faf7362d349ee3ae4b7fcb0719ce0ba3c1`，88 SHA `5b0f3edbb3c641a4826da737016b59f4d6eee1f708b919298a883234d4e2eba7`。

baseline四个hook各1次，matched两个hook各1次；baseline文件后验仍匹配。actual torch-config完整等于旧执行配置。结果SHA `f62eb1d8ae79579ea135afc016290a14143ef4661c8dfccc701a36c28a855b54`；worker结果SHA `b2dfc3788b56fa104661a21813325db535cc0726071cb379577bbd5ccbdd9ab0`。

全部5905个plan bound文件CPU后验通过。before/after实际import+mapped runtime各2743条，实际路径及保留runtime对象字节全部复验。项目helper从封存imports执行，外部库仍从认证安装路径执行；不混淆归档对象与实际导入路径。完整输入/来源身份见旁边evidence.json。

## 资源守卫实际成立

71个样本中memory.max始终10737418240、memory.swap.max始终0、cpu.max始终200000/100000。两PID2123775/2123908均在专属scope、affinity[0,2]，逐PID VmSwap与scope memory.swap.current全0；scope max/OOM/oom_kill/swap fail事件全0。结束后两PID均已退出，已立即向Root释放GPU。

- 递归RSS采样峰值：1,370,578,944 bytes。
- 整GPU采样峰值：3929MiB。
- cgroup memory.current峰值：6,919,081,984 bytes（包含cgroup记账内容，**不等同进程RSS**）。
- host available最低：18,358,583,296 bytes。
- elapsed37.3240411s，无停止原因。

CPU分析v2：6.76s，RSS1,298,020KiB，CPU0/2、BLAS2、CUDA不可见、swap0；两项合成反例测试通过。v1分析保留，v2只增加显式同shape/FP32检查，全部实际数值相同。

## 全张量分母结果

所有75/81/88/out0分母均17,039,360；87分母51,118,080。完整FP32大小、finite和SHA均验证，以下没有依赖六行抽样，也没有从out0倒推官方中间值。

| Native−official | L2 | max abs | NRMSE |
|---|---:|---:|---:|
| 75 |0.104820906393|0.0029296875|2.54465715e−7|
| 81 |0.000271257361|3.337860107e−6|6.65595883e−7|
| 87 |0.007343783343|0.000152587891|1.24210909e−6|
| 88 |0.076925678345|0.000335693359|1.62569943e−6|
| out0 |0.198776544742|0.006591796875|6.01137195e−7|

不同边界尺度和特征宽度不同，不能仅凭跨边界绝对L2比值宣称放大系数。

令N为原native teacher，O为同输入官方baseline，M为官方MLP接受完整native81后得到的对应边界。定义U=M−O（81输入差异通过官方MLP传播）、E=N−M（真正同输入MLP实现差异）。全分母用FP64计算，验证 `N−O = U+E`，atol1e−12。

| 边界 | 总差 L2 | U L2 / max | E L2 / max | cos(U,E) |
|---|---:|---:|---:|---:|
| 87 |0.007343783343|0.010226259108 / 0.000381469727|0.009102706389 / 0.000389099121|−0.717098036815|
| 88 |0.076925678345|0.093101540873 / 0.000389099121|0.082305275053 / 0.000366210938|−0.621480395377|

**这一完整固定teacher输入下，81输入传播与81→87局部实现差异同量级，且明显反向抵消。** 原总差小于其中任一项；不能说误差仅由81继承，不能说MLP局部可忽略，也不能把局部高精度替换必然等同于更接近官方。88的E包含整个同输入MLP差异，绝不是单独down累加舍入误差。

这次完整分母结果支持“当前官方parity误差存在抵消”的事实，但没有证明具体核实现顺序、哪个Gemm或GELU支配、block15是中文step0最早误差源、或正式门槛不可达。前一6行down FP64分解仍仅限其6行；不将它扩大成全张量down误差来源结论。

分析v2 SHA：`d19c2b2900f2355f87348aa1e052cb77cfd143bae1e1efd469fca8ba569d6814`。

## 最小下一边界提议（未实施，未占GPU）

不再修改down。下一项有辨别力的只读观察是**无GELU的up_proj输出84**，在当前已认证native81上对齐native和官方，权重固定为完整认证的gemm_4/up_proj。[4160,12288]完整分母，仍要求native最终out0及官方既有baseline/matched边界全部复现后才解释新增84。

它先把线性投影误差与gate/GELU分支拆开：若84逐位一致，则87局部差异必须在另一个分支或最终乘法中；若84不同，则至少确认ungated Gemm本身贡献实现差异，再根据真实84误差选择下一步。**只有84仍不足以量化它对87的贡献**，因为另一个乘数86尚未取得；不能凭84不同比例给up/gate排序。相比继续调down，这一步测试的是已有全分母证据指向的81→87上游部分，且不把FP64更准当作自动晋级条件。

当前只提交这个具体后续观察建议，未准备新的GPU运行或修改数学。正式中文free-running/质量验收仍未关闭。

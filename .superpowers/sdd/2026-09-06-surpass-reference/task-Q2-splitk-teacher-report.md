# Q2：block15 下投影补偿候选的完整 teacher 诊断

## 当前状态

完整 teacher 已执行，且是有效负结果：候选相对同一官方 out0 的完整 L2/max 均恶化，停止向 step/fullChinese 推进，不修改默认。局部 screen 的通过及历史 compile/invalid-layout 失败见 `task-Q2-splitk-screen-report.md`、提交 `4c35148`；root 的独立复核为 `4d1a2de`。这里不将 screen 结果推广为完整层、完整轨迹或正式质量通过。

## 唯一候选与完整分母

输入沿用已认证 official block14 hidden 与其余九个 conditioning tensor，原始 block15 oracle 不变。仅在固定 param 的 `Gemm gemm_6 1 1 87 88` 将类型替换为 `Q2SplitDown`，其余字节不变；原图 SHA `e8ed770153823bc60b2f27cfac14b9a58f534eb6bd9f19dc692d650f7faddd13`、原始完整权重流 SHA `0321f883d79cab6606ac3d2736344043625f3a0fed9bb1f5fcfdd33d1acd07da` 均强绑定。原始权重不改写、不复制到生产包。

完整输出分母为 4160×4096 = 17,039,360 个 FP32 元素；保存完整 blob87、候选 blob88 和 out0。候选不运行 text/head/scheduler/VAE。不得把这次单层 teacher 称为中文自由运行修复。

独立类继承固定 ncnn `Gemm` 的参数和二进制读入，严格拒绝非 K=12288、N=4096、M=4160、transB=1、alpha/beta=1、constantB=1、无 C、无 quantize 的配置。读出的 B[N,K] 完整转置后，再逐位比较 screen 已认证的全部 50,331,648 个 KN 元素，随后才上传。这样不会只依据模型文件大小猜测矩阵方向或忽略权重流位置。

算法与 screen v3 相同：32 段，每段 384 项；Neumaier FP32 累加并保留 FMA 乘积残差，保留两条 limb，再补偿合并。新增着色器相对 screen 仅增加行起点偏移，不改变每个 dot 的计算顺序。输入显式转换为 FP32 scalar packing，固定检查实际维度；权重使用 flat scalar 上传。

## 分行缓冲与有效性前置条件

每块最多 16 行，临时缓冲 16 MiB，总共 260 块。C++ 逐行 coverage 必须每行恰一次；每块调用 submit-and-wait/reset，保证所有 GPU 读结束后才复用临时缓冲。scalar input、weight、output、temp 的 VkMat 所有者覆盖整个循环。后续残差仍由原图执行。

结果分析前必须同时满足：

1. 完整 blob87 与原 teacher 的 blob87 字节一致，证明改变没有越过下投影边界。
2. 候选 blob88 在固定五行（0、22、4095、4096、4159），全部 20,480 个元素逐位复现 screen v3 的 official-side 输出。
3. 所有 17,039,360 个 out0 元素逐位等于原完整 blob75 + FP32(gate×候选 blob88)，检查残差、packing 和全覆盖。
4. 所有保存 tensor 大小正确、全有限；输入/来源/模型/工具的 bound SHA 在执行前后均不变。

即使以上成立，也仅报告原 teacher/候选对相同 official oracle 的完整 L2/max 及彼此差异，不自动应用正式 gate 或晋级。

## 已完成小验证与封存

独立编译两次 exit0，第一次成功开发入口保留为 `runner-before-weight-identity-check`，第二次增加完整 KN 权重身份检查；两份源码和编译日志均保留。没有改 root CMake 或重新构建生产目标。复用此前冻结的 ncnn 静态库及五个项目数学 wrapper，未从并行 live source 重拷数值代码。

`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p 'test_splitk_*.py'`：6/6 通过。覆盖完整/尾块逐行分母、非法行计划、未知图拒绝、缺失末行拒绝以及先前 FP64 指标分母测试。这些是小 CPU 合同测试，不是 GPU 布局或数值结果。

执行目录：`outputs/q2-block15-splitk-teacher-v1`。
runner SHA：`46237c14f25bfd962d07a7dbc45c5dd3dc71c67242a3fd24126e91e78c37b915`。
plan SHA：`6bdfa3d9a4d117afa621152e01cf48e4e576bd93dd559306ffea31c7a0805369`。
worker 独立身份保存在 `execution/execution-identity.json`，运行入口为封存 `execution/worker.py`，其导入工具也来自该目录。

资源限制：固定 CPU 0、2 两物理核；保留 baseline GPU-block 的内部 num_threads=4，但实际亲和性只有两核，worker 元数据明确二者区别。递归进程树采样 RSS 9 GiB、整 GPU 6144 MiB、host available 至少 3 GiB、2400 s timeout；执行前与过程中均检查资源。GPU 排队期间没有启动候选；得到 paired 明确释放后仅运行一次，现已再次释放。


## 实际执行：有效负结果

session 33651，封存 worker exit0，wall 14.406772078 s；递归进程树采样 RSS 峰 1,358,966,784 bytes，整 GPU 采样峰 3946 MiB，无 guard 触发。这里是采样值，不是驱动分配器精确峰值。日志确认完整 KN 权重 50,331,648 元素逐位一致、4160 行各一次、260 次块提交。完整 blob87 身份、五行 screen 输出和全部 out0 残差重建全部通过。

| 相对同一官方 block15 out0 | 完整 L2 | 最大绝对误差 |
|---|---:|---:|
| 原 native teacher | 0.19877654474212322 | 0.006591796875 |
| 补偿候选 teacher | 0.37041399781155754 | 0.0205078125 |
| 候选与原 native 的差异 | 0.34634726420438156 | 0.01953125 |

候选对 official 的 L2 恶化约 1.863 倍，max 恶化约 3.111 倍。因此先前局部 FP64 点积 screen 的改善没有推广为官方完整层一致性改善。这次没有 v2 screen 那种失效 packing 证据；三项强有效性检查均成立，不能把负结果剔除为未知布局错误。

result SHA：`3ecaa8354d67eac3de9f7a151b0b94c34f569026545d3e482e8732030587f940`。
候选 out0 SHA：`900a97148d7ccea4d700219bee2acc0b44b8da8ba068ad358c24faf39378b6fa`。
候选 blob88 SHA：`c0a5a946c1a94fc95d1203c784c95d5df03a45dbe1296007acc7e4cebaecace1`。
完整未变 blob87 SHA：`a951b1ed772d5a53c0b30ba8a1d1882c52491f6c564cb9f70b09ec6ef4cd3c46`。

## 既存 tensor 的 CPU 分解

原、候选两个 out0 在完整 17,039,360 个元素上均逐位等于同一原生 blob75 加各自 FP32(gate×blob88)。因此该比较中的数值变化确实限定在下投影及其后续正常 FP32 残差传播。

完整 blob88 改变量 L2 为 0.09142308257036884；乘以相同 gate 后的 FP32 更新改变量 L2 为 0.3462013174289788；最终 out0 改变量 L2 为 0.34634726420438156。两者之差（末次加法舍入改变量）的 L2 为 0.01131386064596941。各误差向量并非正交，不能把这些范数相加或当因果贡献比例。

观察结果后才选择的复核行是 652：它同时包含全张量最大误差（列 1568）和最大行误差 L2。对其全部 4096 个下投影输出，用该行真实 blob87 和原完整认证权重重新执行独立 CPU FP64 dot：原 native down L2 0.0021288076077898162、max 0.0008744772772502074；候选 down L2 0.00002567803354400006、max 0.000004726300687707408。即使在这条未属于预声明五行 screen 的最差最终误差行，候选仍明显更接近相同输入的 FP64 dot。该选择是事后审计，不能冒充预声明完整点积分母。CPU wall 1.19 s，RSS 828972 KiB，CPU0/2、BLAS2；第一次辅助脚本路径不存在的 setup 错误/日志保留，改用明确报告目录后成功，没有 GPU 重跑。

原 teacher 误差与候选改变量的 cosine 为 -0.16168045613482313。现有数据仍不能区分“原生上游误差与下投影误差抵消”以及“官方 FP32 内部 reduction 与数学 FP64 oracle 的差异”各自作用；两者也可能同时存在。严禁从 (官方 out0−原生 residual)/gate 倒推出所谓官方 down 真值。

## 唯一下一实验提议，尚未执行

`next-official-hook-proposal.json` 已固定同一十输入、原 stage 执行源、官方权重和原 out0 oracle 身份；SHA `4d54414723d9b6e6ac0c6fe670404e13aa4f5f2acdc53e61b8ee9d3c006e451e`。它是实验提议封存，不是已准备好的执行器或新官方结果。

只运行一个未改公式的官方 `ErnieImageSharedAdaLNBlock` block15，使用原冻结 `export_dit_block.load_block` 路径，不换成导出 wrapper。CUDA FP32、TF32 false、相同原 attention backend；输入 hidden reshape 到原官方 [4160,1,4096]，temb/mask 完全相同。按原 pinned `make_inputs(64,64,64,32,20260905)` 生成旋转角，必须先使派生 cos/sin/mask 全字节匹配已存 in7/in8/in9；不能逆三角函数重建角度。正式执行前还须封存实际 import 来源与最终角度 SHA，当前没有生成新角度或加载新模型。

只注册三个不返回替换值的只读 hook：`adaLN_mlp_ln` 输入对应 blob75、`mlp.linear_fc2` 输入对应 blob87、同模块输出对应 blob88，保留 tensor 所有者并在 forward 后保存全分母。首先要求完整 out0 逐位等于既有 official SHA `a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197`，否则将该 hook 实验判为无效、保存并停止，不能解释中间差异。此后才能用真实官方75/87/88区分上游与下投影差异。没有其他 splitK 调参计划，没有 step/fullChinese 推进。

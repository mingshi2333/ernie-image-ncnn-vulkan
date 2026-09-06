# Q2：官方 block15 真实内部边界与 split-K 负结果分解

## 结论与范围

唯一官方 CUDA block15 forward 已完成；只读 hook 的完整 out0 **逐位复现既有官方 oracle** `a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197`。因此真实75/87/88可以用于分析。没有修改公式、默认实现、正式 gate，也没有继续 splitK 调参、step 或完整中文轨迹。

这批证据解释了局部 FP64 更准但完整官方一致性更差的现象：在明确选择的六行、24,576 个下投影输出中，原 native 与官方各自相对 FP64 的局部舍入误差高度相关（cosine **0.837347**）；补偿候选显著减少自身局部误差，也消除了它与官方舍入误差原先的接近。该分母内，上游传播误差与两侧局部舍入误差的 cosine 仅约 0.021/0.021，不支持“上游误差抵消主导”这一先前未证实解释。不同误差项不是正交的，不能用范数直接表示因果百分比；六行结论也不是全张量 FP64 dot 证明。

## 前置身份与执行

批准提议 SHA：`4d54414723d9b6e6ac0c6fe670404e13aa4f5f2acdc53e61b8ee9d3c006e451e`。使用原冻结 `export_dit_block.py`（SHA `cc2957037b80cce840a0c09e2646c9290d66e7ce6907d57bc14c18978f49a12d`）的 `load_block`，实例化未改公式的官方 `ErnieImageSharedAdaLNBlock`，没有换成 ExportBlock wrapper。官方 ERNIE 源 SHA `0f1814f63008298707afea5d7bd22d0a16073f0a8e858037c198faa28027e926`；官方 block15 权重 SHA `d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2`。

CPU 用原 `make_inputs(64,64,64,32,20260905)` 生成角度，派生 cos/sin/mask 与固定 in7/in8/in9 全字节相同；没有逆三角推算。角度 [1,4160,1,128] SHA `336545ce48a53537d913018c5810a36863b9844c50e7fc093db57d8877f9e465`。其余十输入沿用此前 teacher 原字节，不重新计算 text/head/timestep/latent。完整 hidden 按原官方 [4160,1,4096] 执行。

CPU 前置显式 `CUDA_VISIBLE_DEVICES=-1`，CPU0/2、RSS 3 GiB 上限、host 3 GiB floor、零进程 swap 守卫。v3 前置 wall 9.3679 s、RSS 峰 922,722,304 bytes、无守卫触发。v1 CPU 已通过角度，但 plan 错误纳入尚在写的准备日志；保留原目录且未执行 GPU。v2 修正排除活动日志，事后全部 5472 条绑定无变化；v3进一步把完整 GPU runtime 清单放到 cuDNN 配置解析后、forward前，并将最终 result 写入放到执行后身份复核之后。三版角度相同；只有 v3 执行过 GPU，前两版不是 GPU 失败样本。

实际项目 helper 从封存 `execution/imports/` 导入并验证解析路径。外部 Python 包和运行库**从记录且逐字节认证的实际安装路径执行**，封存目录保留其源/二进制对象；不声称 Python 从这些无扩展名归档对象直接执行。CPU 清单含 2740 个实际导入/映射文件，CUDA 初始化及配置后为 2741，forward后仍为2741；前置和后置均检查原绑定及已加载文件。Python、torch/diffusers 等版本与实际 `.so` 来源完整保存。

实际 CUDA 13.0、cuDNN 92000、TF32 matmul/cudnn 均 false、float32 matmul precision `highest`、官方 `ErnieImageSingleStreamAttnProcessor` 的默认 backend None、eval/no_grad。保留官方 CPU num_threads4，进程实际亲和性只有CPU0/2两物理核。

执行目录 `outputs/q2-block15-official-hooks-v3`；plan SHA `f6aac83a83415bf6eda28b244fa6b16219c4633e9eee5b09ca0e69a57e278519`，封存 runner脚本 SHA `28b44db3c270b52b597a7af6dd4f5d1cbf00c13327726aff44811c9970119785`，worker SHA `fdf27ab6fcf417b25520f6dbf5d2ff412ebbdf3972abc2e592696faf547d70ce`。

session **74926**，唯一 forward，exit0，wall **22.132832654 s**。递归 RSS 采样峰 **1,218,793,472 bytes**，整 GPU 采样峰 **3938 MiB**，host available 最低 **17,126,092,800 bytes**，所有采样进程 swap 为0；无9GiB RSS/6GiB GPU/host3GiB/2400s守卫触发。GPU 已明确释放。以上是采样观测，不是精确分配器峰值或系统 OOM 计数审计。

## Hook 有效性

`adaLN_mlp_ln` 的 pre-hook 保留75，`mlp.linear_fc2` 的 pre-hook 保留87，后者 forward-hook 保留88。每个 hook 恰好一次、返回None，保留 detached 原张量所有者，无原地改写；只有完成原 forward 后才保存。所有边界保存完整FP32分母且全有限：

| 边界 | 元素数 | SHA256 |
|---|---:|---|
| 75 | 17,039,360 | `09189225db6307daa124c88dce40dfc257e8af49d9cea125b44c6c64072b69af` |
| 87 | 51,118,080 | `384328e7033248ef5c43d1a9494d4f93c7e13556bd0913bdda786cfc1f1f10dd` |
| 88 | 17,039,360 | `3a1eb976b4772659e9461745bb21ec9fa92582adfd98d90f728aab83d8375ab8` |
| out0 | 17,039,360 | `a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197` |

最终 result SHA `4d7853827694c2f29cef782d7b2e849e994d4a254bd267ac3e0baf6c136713f2`。先有完整 out0 逐位验证，后有内部数值解释；没有把倒推 target/down 当作官方真值。

## 完整张量比较

所有比较以同一真实官方边界为参照。原native及candidate由此前固定 teacher plan/result 身份再次认证。

| 边界/实现 | L2 | 最大绝对误差 | NRMSE |
|---|---:|---:|---:|
| native75 | 0.1048209064 | 0.0029296875 | 2.544657e-7 |
| native87 | 0.00734378334 | 0.000152587891 | 1.242109e-6 |
| native88 | 0.0769256783 | 0.000335693359 | 1.625699e-6 |
| candidate88 | 0.1059631379 | 0.000930786133 | 2.239359e-6 |
| native out0 | 0.1987765447 | 0.006591796875 | 6.011372e-7 |
| candidate out0 | 0.3704139978 | 0.0205078125 | 1.120201e-6 |

官方、原native、candidate 的全部最终 out0 均逐位重建为各自真实75 + FP32(gate×真实88)。对官方输出，原native残差误差 L2 .1048209、gated更新误差 .1932680、末加舍入项 .0112364；candidate 的残差仍 .1048209，但更新项增至 .3677941、末加舍入项 .0114535。候选恶化已存在于真实88，并由相同 gate 传播，不能归咎于未知残差布局。

## 选定六行的 FP64 分解

行 `[0,22,652,4095,4096,4159]`：包含原五行及事后最大最终误差行652，不是完整51M输入的全行FP64审计。两套真实87分别乘同一完整认证权重，以独立CPU FP64得到 `F_N` 与 `F_O`。定义：

- 上游传播 `U = F_N - F_O`；
- native局部误差 `R_N = N88 - F_N`；
- 官方局部误差 `R_O = O88 - F_O`；
- candidate局部误差 `R_C = C88 - F_N`。

原native对官方差异严格分解为 `U + R_N - R_O`，candidate为 `U + R_C - R_O`；24576元素代数残差均满足独立检查。

| 项 | L2 | 最大绝对值 |
|---|---:|---:|
| U | 0.001805616974 | 0.0000630775761 |
| R_N | 0.003530162151 | 0.000874477277 |
| R_O | 0.003523154533 | 0.000937926674 |
| R_C | 0.0000455688215 | 0.00000683466953 |
| N88−O88 | 0.002704094656 | 0.000156402588 |
| C88−O88 | 0.003925550593 | 0.000930786133 |

`cos(R_N,R_O)=.8373470`，`cos(U,R_N)=.0210226`，`cos(U,R_O)=.0206010`。这些是当前明确分母内的数值关系，不证明两套 GPU kernel 采用同一种累加顺序，也不证明原native具有全模型正确性。它解释了为什么“更接近FP64”不能单独用作本项目官方一致性晋级条件。

CPU最终分析 wall2.39s、RSS1,587,532KiB、CPU0/2、BLAS2，无swap。分析v1与v2均保留；v2补充对历史native/candidate固定plan/result的显式身份认证，数值结论未变。最终 analysis SHA `8a8a9c28f48b311295bd38fbef4fd112bc33b7e15d5252581ef1c33ddce9f844`。

## 检查与停止点

4/4小CPU测试通过：固定完整out0身份、冻结源改写拒绝、全部边界分母、活动准备日志不进入不可变合同。实际运行另验证全部输入/源/权重身份、角度派生、一次hook、全有限、全out0逐位一致及前后runtime身份。没有生产math/CMake改动。当前切片完成；不提出无目标splitK调整，不自动启动任何新GPU或完整轨迹。

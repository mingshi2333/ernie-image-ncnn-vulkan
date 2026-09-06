# Q2 中文1024保存轨迹误差定位

## 完成与边界

两个Vector集成Minor复核关闭，追加提交0361a48。当前新增只读工具`tools/diagnose_text_trajectory.py`与4项synthetic测试；真实分析结果`outputs/q2-chinese-step-comparison-v1.json` SHA cda50efa15d23068f77871037c4c0fed6a31649325fbe136d034e7fc268a6990，工具SHA 07364cb9464f0d4e0a6bb719d1fd744be72491eb6ec270fbc966e0005420414f。

没有完整模型/GPU执行、没有runtime/shader改动。工具先调用现有独立saved-run audit重新核算门槛和所有边界，再逐个绑定文件SHA/shape/finite值。所有计算限CPU2。输出不是正式corpus，也不将差值解释为因果百分比。

## 身份与正确对照

五条轨迹共用官方fixture SHA81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7、package SHA72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1、初始noise SHA f07a2134e662229cd4f6981c648cbbc925080debf514ef51c00d7669dc55e983、同token IDs/config及8步FP32 Vulkan轨迹。

必须保留旧fp32-v1（17/25、PNG23），但不能把它当成已补偿attention下的文本单因素基线。已有kahan/chunked（21/25、PNG13）才是更接近问题的基线；它们runner SHA不同，但保存的text、全部8个prediction/sample、final、decoded逐位相同。新vector runner又不同，因此old/candidate对比仍有二进制版本混杂。

更有控制力的现有对照是chunked与saved official-text：二者runner相同1dd2880274fd862631254966f78e401340393a711be7b27940554715fe13320b，后者条件改为官方text。它把final NRMSE从0.001056088降到0.0003637483，decoded从0.00086949降到0.00030146，但decoded max0.02489112仍失败，PNG max3仍失败。这证明该固定runner下条件误差影响轨迹，也证明仅消除text差异没有闭合剩余误差；不能称所有剩余误差均来自text。

## 首次差异与后期反转

下面是已补偿chunked基线与saved-vector，每步均为零基编号；sample是本步Euler更新后状态。

| step | 基线prediction NRMSE | Vector prediction NRMSE | 基线sample NRMSE | Vector sample NRMSE |
|---:|---:|---:|---:|---:|
| 0 | 0.00010894244 | 0.00013440982 | 5.3475882e-06 | 6.5976131e-06 |
| 1 | 0.00019064445 | 0.00020775212 | 1.3118906e-05 | 1.4642413e-05 |
| 2 | 0.00023436309 | 0.00033489356 | 2.3942558e-05 | 3.1730387e-05 |
| 3 | 0.00062900452 | 0.00059136545 | 7.4797142e-05 | 7.4750863e-05 |
| 4 | 0.0022038825 | 0.00091581169 | 0.00034832062 | 0.00016390708 |
| 5 | 0.0014341728 | 0.00085282251 | 0.00048950634 | 0.0002632399 |
| 6 | 0.0013713675 | 0.0025990208 | 0.00068297262 | 0.00085204485 |
| 7 | 0.001847524 | 0.0020435825 | 0.001056088 | 0.0011538885 |

Vector的text NRMSE从约6.25e-6降低到1.90e-6，但step0在完全相同初始latent下prediction NRMSE更大：1.0894e-4→1.3441e-4。这说明“text更接近”并不保证输出误差单调降低；step0尚无先前latent drift，但文本扰动方向与runner差异仍混在一起，不能归因为某种算子。

最有区分度的后期点是step6：step5后Vector误差L2=0.14072，小于基线0.26167；step6预测误差乘sigma delta后的注入L2却为0.44573，基线仅0.23519。step6后Vector误差反超0.47555（基线0.38119）。step7进一步到0.84916（基线0.77719）。最终Vector decoded NRMSE0.00102033/max0.10383454，比补偿基线0.00086949/max0.09596822更差，虽然通过边界总数22/25高于21/25；PNG仍同样max13。不能只按通过计数宣称整体误差下降。

## 更新方程能排除什么

对每条保存轨迹，以FP64重构
`e_after = e_before + delta * (prediction_native - prediction_official) + residual`。
这里prediction_native使用已经偏离的native latent/text；它与official预测之差不等同于同输入DiT数值误差。

8步schedule的原始linspace值全部是精确二进制分数，shift4和delta采用明确FP32顺序；工具拒绝其他步数，避免假定任意linspace实现一致。Vector step6 residual L2约2.17e-5/max2.926e-7，相对prediction注入0.44573很小。最大step7 residual L2约3.029e-5/max5.419e-7。这支持“直接Euler更新舍入不是当前宏观反转主要量级”；不代表scheduler永远无问题，更不排除timestep embedding、网络残差、Gemm或attention中的误差。

incoming error与本步注入的cosine：Vector step6约0.0611，step7约0.2550，既有误差与新增误差部分同向；不是简单的误差抵消消失可以独立解释所有观察。

## 历史局部诊断的适用范围

- 现有Chinese step0 stage诊断使用相同官方fixture，但runner bf5b7f...与当前不同。其官方输入teacher-force prediction NRMSE约1.3714e-4、max0.00625587，通过该步固定门槛；这不是当前step6的证据。
- 旧step0 input head Vulkan FP32的projected state NRMSE约6.90e-7、调制输出约几e-7，而到block35 image tokens误差约1.70e-4。旧exact-head输入实验仍保留明显block-stack误差。它们提示误差会在块序列中增长，不能据此把当前已补偿attention后的主因指定为Gemm、attention或残差算子。
- `diagnostic-long-step6-fp32-v1`的fixture SHA是ad98e8...，与此次中文81a853...不同。其单步通过不能当作中文step6也通过；报告明确不移植该结论。
- 目前没有匹配当前runner、中文step6、同latent/text/time条件的算子级比较，所以Gemm/残差/timestep“支配”仍是待证假设。纯保存轨迹不能解出这些局部项。

## 下一项实验：当前runner的中文step6同输入teacher-force

先跑一个最小有区分度的实验A，而不是另跑完整8步图像：同一固定包、明确中文fixture、step6输入取官方step5 latent、官方padded-text、官方RoPE/mask，官方timestep feature，当前补偿代码runner快照，36块FP32 Vulkan/direct流程。使用现有工具已支持的入口：

```sh
.venv/bin/python tools/diagnose_pipeline_step.py \
  --model models/turbo1024-s64-portable \
  --reference outputs/pipeline1024-chinese-s64-fp32-v1/reference \
  --step 6 --precision fp32 \
  --runner build-dev/ernie-block-sequence-runner \
  --output outputs/q2-chinese-step6-current-teacher-v1
```

本轮只准备方案，没有执行。执行前先保证最新runner完整构建并记录源码/二进制SHA；工具会再次复制runner。GPU必须等root当前native PE→Vector→512结束后独占串行。现有工具和runner内部写死4个CPU线程，因此若仍要求CPU2，须给进程组设2核亲和性，或者在获分配的probe改动中追加显式threads=2参数；不能谎称现有命令已采用2线程。命令须在新目录执行，保留失败和资源记录。

A的固定门槛仍为NRMSE0.003，max<=0.0002+0.01*official_reference_max（step6 max约5.42092323，阈值约0.05440923）。若A失败，说明同输入情况下现有原生网络/heads数值误差本身足以越门槛，优先定位块内算子。若A显著小于free-running候选误差，说明需要检查latent/text扰动放大和累计效应；通过也不证明局部误差无累计影响。

若A后继续，最有价值的是同runner/同time features的2×2 step6对照：latent取official或saved-vector step5，text取official或saved-vector；四格都绑定相同其他输入，比较prediction的交互项。只有official/official格可以直接作为同输入官方误差门槛；其他格是反事实敏感性诊断，不能用official/official target冒充各格正确oracle。timestep进一步用official与native feature互换须单独控制，不能混入首次文本对照。这样才能区分输入轨迹方向、文本扰动和局部DiT误差，随后才值得选择Gemm/残差/attention内的具体修复。

## 复现本轮只读分析

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_text_trajectory -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python tools/diagnose_text_trajectory.py \
  --run outputs/pipeline1024-chinese-s64-fp32-v1 \
  --run outputs/pipeline1024-chinese-s64-fp32-kahan-v1 \
  --run outputs/pipeline1024-chinese-s64-fp32-chunked-v1 \
  --run outputs/pipeline1024-chinese-s64-fp32-reference-text-v1 \
  --run outputs/pipeline1024-chinese-vectordown-fp32-v1 \
  --output outputs/q2-chinese-step-comparison-v2.json
```

4项synthetic tests通过，覆盖精确Euler注入、独立残差、非有限/shape错配、未审查schedule、hash/size/布尔shape拒绝。五份实际轨迹审计与更新分析全部完成；未改写v1或任何历史tensor。

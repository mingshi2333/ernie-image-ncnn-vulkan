# F2 strength=1 固定1024端点实际结果独立复核

结论：实际执行完整，质量负结果真实。29张量中24通过，5个固定max-abs门槛失败；所有NRMSE及PNG通过。因此不能称strength=1已通过完整质量验收。审查03da492及执行身份修复cfff405；后续合同绑定修复记录见末段。只读CPU12/14、BLAS2，未运行模型/GPU。

## 完整独立重算

证据 `outputs/f2-positive1-1024x1024-v1`。独立检查器/结果在 `outputs/f2-strength1-independent-review-v1/{check.py,result.json}`，每65536元素转FP64累计，完整29项size/shape/有限值/双方SHA和NRMSE/max重算与保存值一致。官方suffix分母为25（6输入、8对prediction/step、3最终）另encoder3+noise，不存在遗漏输出或错用旧strength.5的17/21分母。

| 失败边界 | NRMSE（仍通过） | max abs | 固定max门槛 |
|---|---:|---:|---:|
| prediction3 | .000631872191 | .1090077758 | .06559188385 |
| prediction4 | .000649377051 | .06741416454 | .06381308098 |
| prediction5 | .000878621183 | .08456856012 | .06668334026 |
| prediction7 | .000971264970 | .08887302876 | .07389645119 |
| decoded | .000460751292 | .01358774304 | .01112692375 |

门槛仍为原FP32 NRMSE≤.003、max≤.0002+.01×参考最大值；conditioning保持原.0002/.0002/.0002。没有调整门槛、改变归一化或丢弃失败项。PNG完整RGB1024²独立解码max2、MAE.0051492055257161455、15878通道值不同，满足原max2/MAE.1；PNG通过不能覆盖上述5个tensor失败。

## 输入、源码和实际执行

冻结contract a269a01…、原prompt31字节含LF、9token IDs、输入RGB、noise均与此前准备一致。native initial完整524288元素逐字节等于保存noise，且与sigma=1的native encoder/noise计算逐位吻合。原schedule绝对0..7保持，未只执行后四步。官方encoder边界引用已独立运行88f fixture；本次真实执行官方text/DiT八步/decoder，不声称再次运行官方encoder。

原生249份源码、官方85份源码以及snapshot根lock、runtime三份实际Python源、runner ca2ed9…、模型schema3 manifest2c1d/schema2 manifest72bb全部重算吻合。沿用既存大权重验证链，未再次全扫包所有大对象。真实命令为native CPU text vector/Vulkan FP32 DiT/CPU direct VAE、CPU2 affinity4,6，无PE或embedding注入。

- native exit0，wall808.576175050秒，采样cgroup memory.current峰7384297472字节，host最低12129710080。
- official exit0，wall252.439341491秒，采样峰6410264576，host最低13110706176。
- 两侧10GiB memory.max/swap0实际观测，50ms持续host≥3GiB，timeout1800，OOM/oom_kill均0。这是cgroup采样，不是精确RSS/allocator峰；trace时长不能用于S。没有失败重试被隐藏或用缺失数据造成假负。

## 执行身份发现与修复

原native-plan/identity.json和official-plan/identity-v3.json复用了strength.5，包含旧合同9fd1/start4/计划命令，不能作为当前strength1执行身份。已及时反馈；cfff405新增两份execution/identity.json，绑定当前合同、start0/noise、真实process SHA/command和完整输出清单，旧plan保留为runner/source provenance。它们是执行后整理的身份，不是事前密封声明。

独立验证official40文件identity SHA `cb3ec3fa709bfa43165f7b3f38f7d564240dd812fa12736477c43639ad62b911`、native36文件identity SHA `14292bdd1db898ad2dd4d47650d216199cb8c06a58a015213bc4c5b34eac8a83`，所有文件size/SHA一致，实际命令正确，start/noise SHA逐项与当前合同一致。旧记录没有被覆写。数值失败在身份修复前后不变。

后续73f1c1e将新execution identity中的start/noise分别绑定当前合同SHA，关闭“两者一起改成错误hash仍彼此相等”的缺口；独立14/14 tests通过。修订audit在不覆写原artifact情况下重跑：strength1为quality_gate_failed/29/25，旧strength.5仍pass/21/17。没有剩余阻断解释本次负结果的发现；这份审查关闭证据完整性，不关闭strength1质量缺口。

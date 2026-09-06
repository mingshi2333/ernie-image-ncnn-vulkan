# F1 固定1376×768原生VAE候选准备

状态 `prepared_not_executed`。未执行原生模型、GPU、heads或DiT，未修改生产registry。官方参考实际执行报告6833bad待root独审；本准备也须独审及新的模型槽后才能执行。

使用v5既定冻结`source/tools/specialize_vae.py`，单纯CPU流SHA/图变换退出0；198,481,196B原template权重完整SHA `ee7db4b3725079f9baeb8587bcad56fd3f04d92254b90d86acc894c1ec63cd24`，候选继续指向同一原bin，前后全SHA一致。原图完整SHA `6ecb591c473c56f3d9e923c845be2125e6e42b33ee2cd8d4d1783db68e32c4c5`；恰reshape_99的flatten64→16512、reshape_100的8×8→172×96，其余图字段不变。独立重算特化函数结果与实际graph全文相同。

候选位于`outputs/f1-shape1376-s64-plan-v5/vae-candidate`：param `0476f2736617fbdd20add3dd424773ce556141894cf44d6e04cc328f8adda7dc`；model `50a2bf64ae7d4a5424b0f5115529b4d80b5c32ed3401dab4a4b7457e7b072b70`；conversion `e9ceaa61901df79f1dd3c0c4b18fdff530e443c60a8557a3b9fbb19d0ccda94b`。fixture `dc57109cd986e3bc30cbb99b5a6377470f0649dcef86947b42f25c798c063157`与刚完成官方reference逐字节一致，输入[1,32,96,172]、输出[1,3,768,1376]完整3,170,304元素；保留FP32 atol/rtol=2e-4、NRMSE=2e-5原门槛。模型源/config/revision与模板一致才允许特化。参考输出未复制改数值。

执行准备位于`outputs/f1-native-vae1376-preparation-v1`，569文件大小/fullSHA绑定：v5全部307冻结source、template/candidate完整文件、官方actual证据/worker身份、固定runner、实际validation导入文件/maps、ldd解析的native依赖与ld.so.cache、解释器、time程序和新控制器。导入/ldd是准备范围，不声称独立覆盖所有瞬时dlopen；模型前后复验这些文件。

- plan SHA `57970f0d1afcf7094700d0f56079c24881f790841822e352015b6c2eed29789c`
- controller SHA `98770dd6dacfafd9b7296a4a8b790e2025a123e4a04213d746bc5c6aca8b1b1a`
- launch SHA `0b4c5804090fcf1088a44f011b5530ad07db5bec475560c646973f2d7b9bec56`
- 头runner保持v5 `dbd6d9d057737dc1d7a55f4c4062cc1b85f9653d4de00d2551cc2b212d62caba`，未重编。其CLI decoder width/height≤256，支持本172×96；不等于生产形状注册。

批准后应外部完整核plan/controller/launch SHA，再用launch.json.argv原样启动；只补其中记录的现有user bus环境。启动器要求专属`ernie-native-vae1376-v1.scope`、16GiB/swap0、cpu.max200000/100000、affinity12,14，连续50ms采样host可用≥3GiB、外层1800秒。保留原validator内层900秒runner超时。实际内部ncnn option.num_threads=4；CPU配额2核不是内部threads2。CUDA隐藏。命令固定为同一venv入口+冻结validate_dit_heads.py `--cpu-only --vae-convolution direct`，由该原验证器复制并执行封存runner。

控制器要求execution/validation新目录，不接受复用。每次失败保存validator原exit/log和guard记录；资源失败清理整个专属cgroup中的其他PID，覆盖validator及其另起session的time/native子进程，不仅kill父PGID。成功须唯一CPU/FP32/out0结果通过原gate、实际复制runner SHA一致及全部输入/源前后认证、OOM0。cgroup memory.current采样峰不是精确RSS，不用于S/M；不把通过候选自动注册生产。

CPU小合同4/4通过：相同size输入内容突变拒绝、非专属scope拒绝、跨session PID清理分母、实际小Python子进程exit7原样保留且再次运行拒覆盖。没有用伪模型通过宣称实际正确性。控制器、准备脚本及测试随报告提交，实际冻结副本留outputs；未执行launch.json。

# F2 固定 1024 strength 0.5 实际执行独立复核

结论：本固定开发样本真实原生执行通过。审查 bce1be1 与补丁 edd0ebf；没有剩余阻断本次实际质量结果的发现。此结论不覆盖 formal15/72、任意分辨率/强度、PE、低精度、性能 S/M 或其他平台。

## 独立复算与身份

证据根 `outputs/f2-positive05-1024x1024-v1`。独立检查器与结果在 `outputs/f2-positive1024-independent-review-v1/{check.py,result.json}`，每块65536 FP32元素转FP64累计，未加载模型或使用GPU。17项官方suffix（6输入、4对prediction/step、3最终输出）加encoder mean/packed/normalized和noise，共21项，完整shape/字节数/有限值/双方SHA均核对。全部重算NRMSE与保存结果相符（相对容差1e-12）；最大误差逐值一致。constant-2包含全部4160²元素，decoded包含3145728元素，没有抽样或丢弃失败项。

| 边界 | NRMSE | max abs |
|---|---:|---:|
| prediction4 | 3.5151552e-6 | 7.67707825e-5 |
| prediction5 | 3.0364500e-6 | 4.10079956e-5 |
| prediction6 | 9.1549877e-6 | 7.93635845e-4 |
| prediction7 | 6.5629092e-6 | 3.51190567e-4 |
| final | 4.2310271e-6 | 1.75952911e-4 |
| decoded | 7.7035649e-6 | 2.51993537e-4 |

使用既存pipeline FP32门槛NRMSE≤.003、max≤.0002+.01×参考全局最大值；text/padding/constants使用既存conditioning的.0002/.0002/.0002。encoder在这里使用整链FP32门槛，不能冒充此前独立encoder的更严门槛。PNG独立解码1024×1024 RGB：max1、MAE0.0002568562825520833、808通道值不同，通过原MAE≤.1/max≤2门槛。

原生249份源码与官方85份工具源码逐文件hash复核；官方snapshot根sources.lock也匹配。原生runner `ca2ed9dd12c9e3fc1a4d7b8e0869ef1db51ed20e75b4392801b00e769484c75e`、schema3 manifest `2c1d0cdf39fe4cc94f7133d6dfac97a048123a8ebc8a4a5072e4d5e48368a5ad`、官方schema2 `72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1`均重算吻合。未再次读取所有大权重；这些权重的既存包验证链仍为前提。实际安装pipeline/VAE/scheduler三个源码文件与runtime-identity匹配；该记录不等同于整个Python环境的密封快照。

prompt精确31字节含结尾LF，9个token IDs与trace/official一致。PNG解码RGB、保存noise与trace逐位匹配。native initial按其实际encoder-normalized和相同noise、sigma FP32 .8重新计算，逐位相同；官方initial绑定此前认证官方encoder起点，允许两端encoder浮点误差。encoder使用posterior mode、pixel-unshuffle2/BN eps1e-4，decoder逆BN eps1e-5；本次官方oracle复制已独立执行且固定88f fixture的encoder边界，随后真实执行官方text/DiT后缀/decoder，不能称再次运行encoder。绝对步编号4..7保留原8步schedule后缀。

## 执行资源与失败历史

实际命令为原生文本CPU vector、DiT Vulkan FP32、VAE CPU direct，CPU2/affinity4,6，PE关闭。没有reference/candidate embeddings注入。两份supervisor源码SHA已重算，10GiB cgroup memory.max与swap0实际观测，50ms持续host可用≥3GiB，1800秒超时。

- native exit0，wall494.306705305秒，采样memory.current峰6679691264，host最低11049607168字节。
- official v3 exit0，wall181.194119511秒，采样峰8690458624，host最低10926256128字节。
- 两端OOM/oom_kill均0。memory.current是cgroup采样而非严格RSS峰，GPU独占来自调度记录；此supervisor不提供GPU分配器/整卡峰的强制预算证据。时长含trace，不作为速度比较。

v1路径布局找不到tokenizer、v2缺snapshot sources.lock，均保留非零process与完整日志；没有将它们改写为pass。v3补snapshot根与models链接，模型目录是外部权重来源，不是源码完全隔离。前置身份pending字段保留为冻结时状态，完成情况来自process/result。

## 发现与闭环

bce1be1审计工具未核验JSON内的runner/source实际bytes，资源检查漏memory.max/host最低值/failure。已立即反馈作者；edd0ebf补实际源码、runner、包manifest、runtime源与上述资源检查及负例。独立重跑11/11小tests通过；调用修订audit（不覆盖原artifact）仍pass/21。工具是固定本地证据审计，身份JSON自身不是外部认证根，不能据此接受任意替换的自洽整套证据或宣称密码学签名可信。

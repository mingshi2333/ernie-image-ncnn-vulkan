# Q2 文本真实激活诊断交付（2026-09-06）

**已定位可复核的误差放大位置和下一候选方向，没有修改默认数学，也没有宣称完整图像质量关闭。** 第0层到第24层的官方/原生输出已保存，末层双方分别逐字节重现历史真实PE文本编码结果。最大误差首个显著跃升在0-based第2层的BOS位置MLP；同输入FP64参考进一步表明本次SiLU近似不是主要差异，FP32投影累加尤其down GEMM值得作为独立候选检验。

输入是历史PE增强后的279字符中文，含BOS共315 tokens；native bucket2048。只使用这些真实输入，不按BOS tokenID引入特判。所有官方/原生CPU模型进程串行，最多4threads，没有运行GPU。

## 已实现和测试

- `tools/diagnose_text_stages.py`：独立free-running逐层比较、可选层17blob teacher-forced比较、同输入局部FP32/FP64参考和官方层交叉输入。
- `probes/text_runner.cpp`：仅新增CPU诊断trace入口，支持`--trace-dir`、单模型重复`--trace-blob`、fixture的`--valid-tokens`。不修改src/text_encoder.cpp、norm、attention或正常计算数学。各内部blob使用独立extractor；逐层out0使用与正常路径相同配置，实际最终native文本SHA精确复现验证该诊断入口。
- `tests/test_text_stage_report.py`：最终5/5通过，涵盖shape/finite、signed zero、FP64差异不被FP32舍入掩盖、teacher/free分离、blob producer校验。Python编译检查通过。
- 构建仅`cmake --build build-dev --target ernie-text-runner -j2`，与其他工作者协调后执行，session22943 exit0。没有使用旧build/拷贝冒充新runner。
- 初次工具提交`0260c05`；后续局部交叉/FP64扩展另commit。实际执行的脚本和runner都先复制快照并记录SHA，worker实际执行snapshot，不导入live本项目tools。PyTorch/transformers class、config、weight和输入身份记录在每个result。

## 精确第2层blob对应（其它被检查层同一已校验图契约）

| 阶段 | blob | 布局 |
|---|---|---|
| input norm / Q / K / V | 6 / 10 / 13 / 16 | [1,tokens,width] |
| Q RoPE / K RoPE | 45 / 58 | [heads,tokens,128] |
| SDPA / flatten / o_proj | 59 / 61 / 62 | heads CHW / [1,tokens,4096] / [1,tokens,3072] |
| attention residual / postnorm | 63 / 66 | [1,tokens,3072] |
| gate / SiLU / up / gated multiply | 69 / 70 / 71 / 72 | [1,tokens,9216] |
| down / output | 73 / out0 | [1,tokens,3072] |

读取param producer类型并用固定package param hash限制对应关系；以显式shape检查，不按元素总数相同强制reshape掩盖布局差。

## 逐层证据

真实embedding逐位一致。YaRN cos和sin最大差各5.960464477539063e-8。第0层使用official常量的teacher out NRMSE3.1075963e-6，native常量free out3.1075984e-6；这组结果不支持YaRN是主差异来源。

| 层（0-based） | free NRMSE | 最大绝对差 | 最大差位置 |
|---|---:|---:|---|
| 0 | 3.1075984e-6 | 0.00002074242 | token0/channel2 |
| 1 | 3.6072665e-6 | 0.00002241135 | token0/channel2 |
| 2 | 6.3784878e-6 | 0.00384521484 | token0/channel0 |
| 3..23 | 约5.96e-6..6.39e-6 | 0.00384521484 | 同一主通道 |
| 24 | 5.9998705e-6 | 0.00378417969 | token0/channel0 |

官方第1层最大值4.1205683；第2层token0/channel0为592.3843384，native为592.3804932；第24层官方约598.90399。高幅通道在MLP出现后，早期差异长期保留。

末层官方SHA `2ef687ce362212b6dec95f953f7465d2131972b072ab7ed3e5b2ee39e420d01e` 与历史oracle/reference/text.f32相同；native SHA `9270e19b0c59d7b2cff851c73d6a034be0c696237f34517520afa08e637dd426` 与历史trace/text.f32相同。

## 第2层同输入teacher与交叉

同一个官方第1层输出和official cos/sin送入native第2层，out仍为NRMSE3.1519817e-6/max0.00170898438。teacher阶段最大差从postnorm5.6266785e-5，增至gate8.7738037e-5、up0.00024032593、gated乘积0.01440429688，最后down/output0.00170898438。

进一步运行**官方第2层(native第1层输出)**：

| 比较 | NRMSE | 最大差 |
|---|---:|---:|
| native层(official输入) vs official层(official输入) | 3.1519817e-6 | 0.00170898438 |
| official层(native输入) vs official层(official输入) | 8.5212608e-6 | 0.00506591797 |
| native层(native输入) vs official层(native输入) | 2.3536538e-6 | 0.00122070313 |
| 完整native层路径 vs 完整official层路径 | 6.3784878e-6 | 0.00384521484 |

交互项max0.00048828125。native free路径用native常量，teacher/official用official常量；已测常量差很小，但这里“decoder差”仍包括该常量实现差异，不能称纯单算子因果隔离。误差有抵消：official层吃native上游输入的误差比实际native路径更大，不能做线性百分比分配，也不能只追求一个局部指标变小。

## 同一真实native阶段输入，独立FP64局部参考

Norm使用已保存native attention-residual；gate/up用完全相同native postnorm；SiLU用相同native gate；down用相同native gated乘积。各行因此隔离该算子的本地误差，而没有把上游输入差算到当前GEMM。

| 算子 | native 对 FP64 NRMSE | Torch FP32 对 FP64 NRMSE | native / Torch FP32 对 FP64 最大差 |
|---|---:|---:|---:|
| postnorm | 5.4635539e-7 | 6.2775224e-8 | 2.5419887e-5 / 2.0216161e-6 |
| gate GEMM | 1.0965932e-6 | 2.4995726e-7 | 2.0146542e-5 / 4.5652065e-6 |
| up GEMM | 1.0409125e-6 | 2.5146957e-7 | 0.00010479022 / 9.4227894e-6 |
| SiLU | 3.8215814e-8 | 3.8206764e-8 | 7.2031578e-7 / 7.2031578e-7 |
| down GEMM | 2.9500389e-6 | 1.1335291e-7 | 0.00176681668 / 0.00006423801 |

SiLU native/Torch FP32直接比较max1.1920929e-7，几乎同一准确度；不支持把本例主误差归为SiLU近似。down投影差异较显著，特别是BOS高幅真实输入；建议下一步评估**一般文本FP32 GEMM累加策略**，以实际gate/up/down输入为最小regression，再做完整25层和图像回归。不能用tokenID特判，也不能假设采用FP64一定改善完整轨迹。

**旧FP64 RMSNorm负结果仍有效。** 本次只读FP64 norm用来定位，不重跑或重新包装为已接受修复。主运行默认数学没有变化。

## 证据与运行状态

- `outputs/q2-text-stages-layer0-v1/`：第0层单层和17blob；session63113 exit0。
- `outputs/q2-text-stages-full-v1/`：25层演变、双方历史byte复现；session14496 exit0。
- `outputs/q2-text-stages-layer2-v1/`：前3层和第2层teacher；session11455 exit0。
- `outputs/q2-text-local-layer2-v1/`：官方交叉及FP64局部参考；session44650 exit0，实际worker约3.77秒。
- 每个运行保留snapshot、run.json命令/退出码/外部时钟、权重/input hashes、文本tensor以及局部FP64必要row0。没有保存完整attention矩阵或全部FP64大权重数组；仅进程内暂存当前单层和投影所需FP64数据，运行结束释放。
- 最后扩展了后续执行的完整`source_inventory`和严格dtype/bytes bitwise判定；历史执行各自snapshot保留。上表实际结果来自对应运行snapshot，不冒称旧结果来自最终工具版本。

该工作为诊断，尚无默认数学修复或完整PE质量关闭。官方text注入曾让decoded最大差通过，但final/decoded NRMSE并未整体下降，不能据此写“文本是唯一原因”或“全部链路误差改善”。root保留该图像级结果与负面变化。peer下载仍由root监督，本任务没有重启下载。

## 2026-09-06：改变 down 归约顺序的两项候选

**InnerProduct 二维 batch 图替换为负结果。** 固定图的 Gemm 为动态 A、transA=0、constant B、transB=1、C broadcast=-1、alpha/beta默认1。上游 Gemm 读取 `(K,N)`，InnerProduct 读取 `N*K`，不带bias，两者消耗同一权重流；本包为BF16标记的无损磁盘存储，FP32运行时展开。仅替换第2层 down，原bin硬链接、SHA相同，其他图行不变。真实teacher输入下 gated72、down73、out0 全部逐字节等于原Gemm（session98251 exit0，外部4.875秒，含多次trace提取，不是投影计时）。同一输入的FP64差当然不变，不重复全25层。`innerproduct_gemm_fp.h`二维路径的多个sum对应不同batch/output，并非对同一点积K分段，仍是长FMA链。

`outputs/q2-innerproduct-layer2-v1/`保存派生图、runner、prepare snapshot、输入/权重身份与运行记录。`layout-v2/result.json`补充M=256/2048、K=3072、N=16的稀疏选列小型契约：Gemm/InnerProduct四项均与精确预期逐字节相等，CHW输出为[1,M,16]；只证明二维布局。第一次`layout/`因当前build未启用Noop而失败，保留失败log；改为Reshape后通过，不删失败。

**InnerProduct 一维逐行路径取得单算子局部收益，尚未验证完整25层。** 新`ernie-text-gemm-diagnostic`只加载一次真实down权重，每次将一行[9216]作为dims=1输入，315行串行，ncnn CPU2线程。`innerproduct_fp.h`向量路径按K使用多个partial sums并归并，真实改变归约次序。没有tokenID特判，没有修改ncnn源或默认runtime。

| 同一真实gated输入，对FP64投影 | NRMSE | 最大绝对差 |
|---|---:|---:|
| 原Gemm batch | 2.9500388839e-6 | 0.0017668166795 |
| InnerProduct vector逐行 | 6.1454721144e-7 | 0.00036300808574 |
| Torch FP32 | 1.1335290585e-7 | 0.000064238008008 |

vector对Torch FP32最大差0.00042724609375。vector仍比Torch FP32误差大，不称最优或质量关闭。解包后的每一个down权重FP32值与官方safetensors完全一致；原封不动的gated输入SHA `d04861b7da004d0181360a6f7477f3677fdcf4d7c4eaa8fb21a425393c28a347`。runner SHA `a71f362569a7b7c2241c886231a918eba57f776864ad9804eaf56eb518c30af6`。CPU记录load0.161125秒、compute+I/O0.591890秒、进程peakRSS229220KiB；外部wall0.824707秒；session37967 exit0。该计时没有与相同scope的Gemm计时配对，不能给speedup结论。

证据`outputs/q2-innerproduct-vector-layer2-v2/{identity.json,result.json,process/,prepare.py.snapshot,evaluate.py.snapshot}`。首次`...-v1/`提取错误地假设磁盘FP32标记，assert失败且未运行，保留零字节目标；v2正确使用BF16标记，随后与官方完整矩阵逐元素检查。构建第一次因CMake尚未重新生成新target失败，保存`q2-text-gemm-diagnostic-build-v1.log`；配置后`-j2`单target成功，日志v2。下一步需要把同一逐行策略放入仅诊断的完整25层路径；本报告时未执行，未晋升默认。

## 2026-09-06：完整25层 down-only 诊断实现

新增仅`ernie-text-runner --diagnostic-vector-down`可注册的`DiagnosticVectorDown`，仅在CPU trace路径可用。其包装上游`InnerProduct`、加载一次当前层权重，并对全部行（包括padding）调用dims=1路径；没有prompt、tokenID或valid-row特判。计算线程固定2。普通图和默认运行时没有改动。`derive_vector_down_graph`只允许已核验的固定2048图中唯一`gemm_6`（72→73、K9216/N3072、无bias、transB=1、无C）替换，未知参数、重复层、非2048图均拒绝。原始param/bin逐项验证schema2 manifest SHA；派生param有独立SHA，bin字节不变。

`outputs/q2-text-vector-wrapper-layer2-v1/`在实际2048行第2层teacher fixture执行三个独立extractor（gated、down、out0），exit0；315个有效行的gated输入SHA与baseline完全相同，down结果SHA与独立vector runner完全相同。这是包装一致性验证，不是新的性能benchmark。

新7项测试通过：原5项诊断指标/producer/finite/bytes契约，C++ vector runner实际精确选列及失败处理，以及graph转换的严格参数/唯一性测试。普通pytest运行若未设置`ERNIE_TEXT_GEMM_DIAGNOSTIC`，C++集成项会显式skip；本次设置实际build-dev runner路径并执行通过。构建`ernie-text-runner -j2`独立target成功，日志`outputs/q2-text-vector-down-build-v1.log`。无ncnn上游修改。

完整运行`outputs/q2-text-vector-down-full-v1/`的runner SHA `29c2aeee8a02025705d86c9766acc1c81e6e0424413508218e3aec641e6be94e`。冻结真实PE IDs SHA `d38a9286f9cb2478213f1539678b0a288142ca13c92113044a748b90700a0814`，与前述baseline相同。每100ms检查host MemAvailable至少3GiB和进程RSS不超过2GiB；任一触发则终止并保存失败。每层Net离开作用域后释放，未同时驻留整套文本权重。完整source inventory额外记录为运行期间观察，明确其他worker文件可能在binary构建后变化，不能冒称全仓库是一次冻结build；本probe源snapshot、runner字节、原/派生graph和权重身份在启动前固定。

### 已执行的完整结果

session56912 exit0；25层均完成。外部wall200.526630532秒（包含trace和I/O，非paired benchmark），10Hz采样峰值RSS1189488KiB，最低MemAvailable4631764KiB。未触发资源停止；采样峰值不能保证捕获所有瞬时峰值。

| layer | 原NRMSE | down-vector NRMSE | 原max | down-vector max |
|---:|---:|---:|---:|---:|
| 0 | 3.10759836e-06 | 2.82896574e-06 | 2.07424164e-05 | 1.74045563e-05 |
| 1 | 3.60726652e-06 | 3.40823594e-06 | 2.24113464e-05 | 1.90734863e-05 |
| 2 | 6.37848776e-06 | 1.59009794e-06 | 0.00384521484 | 0.000915527344 |
| 3 | 6.37186337e-06 | 1.61677426e-06 | 0.00384521484 | 0.000915527344 |
| 4 | 6.36500916e-06 | 1.64467137e-06 | 0.00384521484 | 0.000915527344 |
| 5 | 6.36100836e-06 | 1.68933036e-06 | 0.00384521484 | 0.000915527344 |
| 6 | 6.36268176e-06 | 1.71715947e-06 | 0.00384521484 | 0.000915527344 |
| 7 | 6.36044797e-06 | 1.73699469e-06 | 0.00384521484 | 0.000915527344 |
| 8 | 6.37027707e-06 | 1.781648e-06 | 0.00384521484 | 0.000915527344 |
| 9 | 6.37294054e-06 | 1.81225685e-06 | 0.00384521484 | 0.000915527344 |
| 10 | 6.37423632e-06 | 1.82877025e-06 | 0.00384521484 | 0.000915527344 |
| 11 | 6.37872108e-06 | 1.83659857e-06 | 0.00384521484 | 0.000915527344 |
| 12 | 6.37346025e-06 | 1.85044764e-06 | 0.00384521484 | 0.000915527344 |
| 13 | 6.38650154e-06 | 1.87123004e-06 | 0.00384521484 | 0.000915527344 |
| 14 | 6.38109982e-06 | 1.88004146e-06 | 0.00384521484 | 0.000915527344 |
| 15 | 6.36387241e-06 | 1.86465494e-06 | 0.00384521484 | 0.000915527344 |
| 16 | 6.34914273e-06 | 1.8784348e-06 | 0.00384521484 | 0.000915527344 |
| 17 | 6.29522345e-06 | 1.85915218e-06 | 0.00384521484 | 0.000915527344 |
| 18 | 6.25118243e-06 | 1.8862663e-06 | 0.00384521484 | 0.000915527344 |
| 19 | 6.20668362e-06 | 1.91319153e-06 | 0.00384521484 | 0.000915527344 |
| 20 | 6.11482961e-06 | 1.99674678e-06 | 0.00384521484 | 0.000915527344 |
| 21 | 6.05411179e-06 | 2.2108909e-06 | 0.00384521484 | 0.000915527344 |
| 22 | 6.00671711e-06 | 2.45349938e-06 | 0.00384521484 | 0.000915527344 |
| 23 | 5.96145878e-06 | 2.74271987e-06 | 0.00384521484 | 0.000915527344 |
| 24 | 5.99987048e-06 | 3.13618865e-06 | 0.00378417969 | 0.000915527344 |

最终文本NRMSE从5.9998704804e-6降到3.1361886513e-6，max从0.0037841796875降到0.00091552734375。`output.f32`为315×3072有效行FP32，SHA `433ae46eaa1c8b831fa273f68b8345646d6ddfdb56695b4d3286b4964bca294a`。所有层max/NRMSE相对原文本改善；这是一份真实PE增强prompt的文本证据，不能外推所有prompt或宣称图像门槛已经关闭。root将安排同初始噪声/轨迹的图像闭环。

历史official最终SHA `2ef687ce362212b6dec95f953f7465d2131972b072ab7ed3e5b2ee39e420d01e` 和原native最终SHA `9270e19b0c59d7b2cff851c73d6a034be0c696237f34517520afa08e637dd426` 与先前artifact完全一致；候选更改与比较对象身份明确区分。`result.json`记录每层三方SHA及原/候选/官方三方指标，不覆盖任何历史结果。

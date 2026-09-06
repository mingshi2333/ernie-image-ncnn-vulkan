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

## 保存候选文本的图像诊断入口（本子任务未运行GPU）

`tools/validate_pipeline.py --diagnostic-embeddings PATH`只接收明确声明的原始little-endian FP32 `.f32`存储，shape绑定已校验官方fixture的 `[1,len(ids),3072]`，严格检查字节数与finite；复制前后source及本地snapshot的SHA必须相同。裸float文件没有自描述dtype，因此dtype是CLI的显式存储契约，不能把文件后缀当作自描述元数据证明。实际runner的`--embeddings`只指向输出目录的`diagnostic-embeddings.f32`；执行后的trace/text必须与冻结snapshot精确SHA相同。该模式与PE、reference-embeddings、reference-only互斥，仍可重用完整官方`--reference`，固定原有门槛。

结果明示`conditioning_source=saved_candidate_diagnostic`、`native_acceptance_eligible=false`及原生text bypass。`diagnose_trajectory.py`额外核验source scope、dtype/shape/tokenIDs、snapshot散列与实际command路径（拒绝外部源直读或symlink），以及实际text trace；text边界输入身份包括候选embeddings SHA。未知scope或伪造资格/元数据拒绝。既有原生/官方text诊断路径保留。

6/6轨迹单元测试通过，覆盖新快照精确读取、实际trace/command/shape/dtype/tokenID篡改、FP64错误字节数、NaN、四种互斥选项；py_compile和diff检查通过。既有PE增强prompt文件与官方fixture prompt字符串精确相同、315 IDs、text shape [1,315,3072]已只读确认。这里仅完成接口和小测试，实际DiT/VAE图像执行交由root；不能把实现完成写成图像验证完成。

## 历史中文32-token与长英文40-token：完整25层独立诊断

两份输入均来自历史development失败轨迹，没有使用正式72-case corpus。中文32是有效token数，静态bucket为64；英文40有效tokens同样bucket64。package是原`models/turbo1024-s64-portable`，manifest hash与两份历史result完全相同，25层param/bin、embedding/frequencies逐文件检查manifest SHA。使用此前固定runner `29c2aeee8a02025705d86c9766acc1c81e6e0424413508218e3aec641e6be94e`；原图baseline也走诊断CPU2线程但图内没有custom down，候选只把各自64静态图的down替换为同一逐行wrapper，全部64行执行。

历史只保存最终official text，因此新增一block一block的CPU2官方重放以补逐层值，并强制最后一层SHA与历史完全相同；原生原图同样强制完整最终SHA与历史完全相同，失败即停止。两份样本都满足这两项精确复现。

### chinese: 32有效tokens / bucket64

提示词：雪山脚下的蓝色湖泊，松树林，清晨阳光，写实风景摄影。

IDs SHA `cfa74e72689e34d15df98c1c76c137e831e7f08d78dc9f358eb1e0da1abf07f1`；prompt UTF-8 SHA `d778f6ec02a9bfdf47dcaa14b869574a110876e9cca248678235f440f23c18f1`。

历史official SHA `20be68ebb735bc11c61a4093327659529a656e38d25e7fa13fa5b1c6dedce0ac`；历史native SHA `8c8a2261fd6c13f2771d5dc52329d97ac9129daaaca38dbf86b6458f4fd2ff8e`。候选`outputs/q2-history-vector-down-v2/chinese/candidate.f32` SHA `c9eef032e54fe6a2a0d3eeb13458f66821a3442308684b330fa2ca72032c0138`。

| layer | 原NRMSE | down-vector NRMSE | 原max | down-vector max |
|---:|---:|---:|---:|---:|
| 0 | 3.42066518e-06 | 3.09208317e-06 | 2.07424164e-05 | 1.74045563e-05 |
| 1 | 4.28104138e-06 | 3.99101524e-06 | 2.24113464e-05 | 1.90734863e-05 |
| 2 | 6.39691446e-06 | 1.55021613e-06 | 0.00384521484 | 0.000915527344 |
| 3 | 6.39348957e-06 | 1.55137676e-06 | 0.00384521484 | 0.000915527344 |
| 4 | 6.39047354e-06 | 1.5544289e-06 | 0.00384521484 | 0.000915527344 |
| 5 | 6.38736539e-06 | 1.5603598e-06 | 0.00384521484 | 0.000915527344 |
| 6 | 6.38670878e-06 | 1.5649258e-06 | 0.00384521484 | 0.000915527344 |
| 7 | 6.38660292e-06 | 1.57007632e-06 | 0.00384521484 | 0.000915527344 |
| 8 | 6.3867719e-06 | 1.57400187e-06 | 0.00384521484 | 0.000915527344 |
| 9 | 6.38550088e-06 | 1.57754813e-06 | 0.00384521484 | 0.000915527344 |
| 10 | 6.38492756e-06 | 1.58071167e-06 | 0.00384521484 | 0.000915527344 |
| 11 | 6.37985005e-06 | 1.58032274e-06 | 0.00384521484 | 0.000915527344 |
| 12 | 6.37621742e-06 | 1.5800643e-06 | 0.00384521484 | 0.000915527344 |
| 13 | 6.37397593e-06 | 1.57860812e-06 | 0.00384521484 | 0.000915527344 |
| 14 | 6.37173042e-06 | 1.5776651e-06 | 0.00384521484 | 0.000915527344 |
| 15 | 6.36587494e-06 | 1.57469436e-06 | 0.00384521484 | 0.000915527344 |
| 16 | 6.36155321e-06 | 1.57613301e-06 | 0.00384521484 | 0.000915527344 |
| 17 | 6.33863234e-06 | 1.57347618e-06 | 0.00384521484 | 0.000915527344 |
| 18 | 6.33438161e-06 | 1.58133395e-06 | 0.00384521484 | 0.000915527344 |
| 19 | 6.32991379e-06 | 1.58713698e-06 | 0.00384521484 | 0.000915527344 |
| 20 | 6.31904674e-06 | 1.60681009e-06 | 0.00384521484 | 0.000915527344 |
| 21 | 6.31473882e-06 | 1.64579543e-06 | 0.00384521484 | 0.000915527344 |
| 22 | 6.31387665e-06 | 1.69783561e-06 | 0.00384521484 | 0.000915527344 |
| 23 | 6.3114183e-06 | 1.76866725e-06 | 0.00384521484 | 0.000915527344 |
| 24 | 6.25437591e-06 | 1.90232809e-06 | 0.00378417969 | 0.000915527344 |

| 阶段 | 外部wall秒（含启动/trace/I/O） | 10Hz峰值RSS KiB | 最低host可用KiB |
|---|---:|---:|---:|
| official | 15.044949747 | 1430680 | 7079168 |
| baseline | 19.967350186 | 615916 | 6796848 |
| candidate | 24.894669689 | 619008 | 6768560 |

### long: 40有效tokens / bucket64

提示词：A small white cat sitting beside a blue ceramic teapot on a wooden desk, warm afternoon sunlight through a window, a green plant in the background, soft shadows, realistic photograph with fine detail.

IDs SHA `dfc478b4af2fa60de5f9f90ed9a6ae3ec7c7a8a83316a811b310bed4ca05ca1a`；prompt UTF-8 SHA `72062c4cea8a89f2425e631787b72c5c8fe8f2c7e15020dbbb5f7b22c56be37e`。

历史official SHA `e65d6d26169ff783b93080a395cd2b56dbbe80472100a80206c815768f9e8019`；历史native SHA `ed359c938f4d224674788d42c8ac7e34acf74b18e108018a8382ba2c6352c662`。候选`outputs/q2-history-vector-down-v2/long/candidate.f32` SHA `fcc79b86564cf3afc1db1829a530256728cbc188744e6d44a248bc4ddb039c40`。

| layer | 原NRMSE | down-vector NRMSE | 原max | down-vector max |
|---:|---:|---:|---:|---:|
| 0 | 3.20370754e-06 | 2.82645494e-06 | 2.07424164e-05 | 1.74045563e-05 |
| 1 | 3.46256139e-06 | 3.21067689e-06 | 2.24113464e-05 | 1.90734863e-05 |
| 2 | 6.39578246e-06 | 1.54799948e-06 | 0.00384521484 | 0.000915527344 |
| 3 | 6.39260901e-06 | 1.54976394e-06 | 0.00384521484 | 0.000915527344 |
| 4 | 6.38934011e-06 | 1.55366183e-06 | 0.00384521484 | 0.000915527344 |
| 5 | 6.38581745e-06 | 1.56140466e-06 | 0.00384521484 | 0.000915527344 |
| 6 | 6.38464896e-06 | 1.56455359e-06 | 0.00384521484 | 0.000915527344 |
| 7 | 6.38344701e-06 | 1.5694641e-06 | 0.00384521484 | 0.000915527344 |
| 8 | 6.3835366e-06 | 1.57542107e-06 | 0.00384521484 | 0.000915527344 |
| 9 | 6.38266057e-06 | 1.58128743e-06 | 0.00384521484 | 0.000915527344 |
| 10 | 6.37380013e-06 | 1.58114193e-06 | 0.00384521484 | 0.000915527344 |
| 11 | 6.36435249e-06 | 1.58191354e-06 | 0.00384521484 | 0.000915527344 |
| 12 | 6.36047607e-06 | 1.58371154e-06 | 0.00384521484 | 0.000915527344 |
| 13 | 6.35943098e-06 | 1.58441727e-06 | 0.00384521484 | 0.000915527344 |
| 14 | 6.35843203e-06 | 1.58586415e-06 | 0.00384521484 | 0.000915527344 |
| 15 | 6.35295503e-06 | 1.58559012e-06 | 0.00384521484 | 0.000915527344 |
| 16 | 6.34807977e-06 | 1.59129332e-06 | 0.00384521484 | 0.000915527344 |
| 17 | 6.32467386e-06 | 1.58916429e-06 | 0.00384521484 | 0.000915527344 |
| 18 | 6.31906816e-06 | 1.59695028e-06 | 0.00384521484 | 0.000915527344 |
| 19 | 6.313543e-06 | 1.60505058e-06 | 0.00384521484 | 0.000915527344 |
| 20 | 6.31286698e-06 | 1.63224611e-06 | 0.00384521484 | 0.000915527344 |
| 21 | 6.30352024e-06 | 1.67369618e-06 | 0.00384521484 | 0.000915527344 |
| 22 | 6.29901407e-06 | 1.72876402e-06 | 0.00384521484 | 0.000915527344 |
| 23 | 6.27511636e-06 | 1.80284572e-06 | 0.00384521484 | 0.000915527344 |
| 24 | 6.19127602e-06 | 1.94835703e-06 | 0.00378417969 | 0.000915527344 |

| 阶段 | 外部wall秒（含启动/trace/I/O） | 10Hz峰值RSS KiB | 最低host可用KiB |
|---|---:|---:|---:|
| official | 16.453476191 | 1438948 | 6337748 |
| baseline | 20.168572320 | 609868 | 6515096 |
| candidate | 24.893488307 | 604116 | 6475892 |

两样本所有25层的NRMSE/max都改善。中文最终6.254375907e-6→1.902328094e-6；英文6.191276017e-6→1.948357034e-6；两者最终max均从0.0037841796875→0.00091552734375。相较前述被拒绝的FP64 RMSNorm候选，本次确有两个独立历史提示词的完整文本收益，但仍不是图像质量关闭或正式corpus验收。候选wall约24.9秒、原图约20秒只作运行事实；存在并行root工作、trace和加载，不能据此提供受控speedup结论。

运行`outputs/q2-history-vector-down-v2/`，session64705 exit0，官方/原生/候选六个子进程全部exit0，始终CPU2、无GPU。每100ms监控host可用≥3GiB、RSS≤2GiB；未触限。采样峰值不是精确瞬时峰值保证。source inventory、独立worker/runner snapshots、每阶段命令/退出码/时钟和输入/输出SHA齐全。

首次v1/session72986因parser不允许stage-layer=-1（仅逐层输出）在模型加载前exit2，保留日志；随后给官方worker明确允许-1，其他路径仍拒绝。增加`--threads 2`控制官方CPU线程，并在选择性stage捕获结束后删除module字典/局部引用，避免跨层持有前一个block权重。图转换只接受显式64/2048桶；64图没有改写任何static reshape。新增64显式参数与未审桶拒绝测试，7/7文本诊断tests通过。旧RMSNorm负结果、所有历史图像失败与固定门槛保持。

## 可选原生 TextDownMode::Vector 组件

CPU `run_text_blocks(..., stats, TextDownMode down_mode=TextDownMode::Gemm)`新增兼容尾参。默认路径原有图、权重读取与数学不变；Vulkan签名不变。Vector仅CPU FP32，完整图token模板逐项审查独立64/2048导出、包括全部58层/75blob和所有reshape/连接/参数。只有完全匹配才在内存把唯一down Gemm改成ErnieTextDown；源模型包文件不改写。32桶和未知图显式拒绝vector，默认Gemm仍可用。

新`src/ernie_text_down.*`注册ErnieTextDown，固定K9216/N3072、无bias、无activation/量化附加参数。加载前流式检查完整block两个rawFP32 norm段及七个tagged矩阵、标签/长度/EOF，支持原FP32和无损BF16磁盘标记，未知tag、截断/附加数据拒绝；只读少量标签并seek，不复制整层权重。自己的down从同一ModelBin读取N*K一次，检查展开dtype/shape/finite，交给InnerProduct用Mat引用，不再读取或重排磁盘数据。逐全部行调用pinned ncnn dims1内核，不按valid/prompt/token特判，无中间tensor文件I/O。

probe新增`--text-down-vector`，必须CPU且无诊断trace，实际调用上述生产run_text_blocks；普通模式也可用`--threads 2`验证相同线程条件。原diagnostic wrapper路径保留为独立参考。CPU contract检查完整64/2048图、非审查桶、图边/维度/参数篡改，bias/未知param/错误参数dtype，完整FP32/BF16结构流的EOF/tag/truncation，错误weight形状/NaN，真实固定维度稀疏矩阵逐行精确oracle、错误输入宽度/finite/低精度。CTest `text_down_contract_cpu` 1/1通过（最新0.36秒）。两个静态param fixture已纳入test，仅几KB，不带模型权重。

真实native runner SHA `91ec9c70d04c0eaf14277076177941d64d4c875b4dc71fc25274265f8d19c452`；`outputs/q2-native-text-vector-v1/`保存实际源snapshot、完整source inventory、原包所有text权重/图的SHA、输入身份、运行command/log与比较。session93892 exit0，四个子进程均exit0，未运行GPU。

| fixture / mode | byte-match既存目标 | load秒 | compute秒 | 外部wall秒 | 10Hz峰值RSS KiB | 最低host可用KiB |
|---|---|---:|---:|---:|---:|---:|
| chinese / gemm | True | 39.166913019 | 3.576999061 | 45.183484715 | 623256 | 4827952 |
| chinese / vector | True | 45.748896572 | 14.426441273 | 63.012557066 | 620664 | 3988748 |
| long / gemm | True | 42.073102681 | 3.869200606 | 48.801281405 | 616736 | 4711516 |
| long / vector | True | 44.554350582 | 14.456175798 | 61.897880032 | 610608 | 4254448 |

默认Gemm的目标是历史native baseline；Vector的目标是此前独立diagnostic candidate。两者中文/英文全部SHA逐位相同，因此本原生封装保留已测完整文本数值结果。仍每次一个block、2线程，host可用<3GiB或RSS>2GiB会停止，未触限。采样峰值不保证捕获所有瞬时峰值。

单独执行25个真实bin的结构审核，总观察0.07627811秒，`structural-audit-cost.json`记录每block及test runner SHA；该数字**不含**权重读取/解码、down finite扫描与kernel packing。上表load包含这些全部工作，不能把两种模式的load差因果归给结构审查；并行root作业与缓存状态不同，本批明显load占主导。Vector计算时间也较Gemm高，精度收益有计算成本，未提供speedup/正式性能结论。

本次只实现可选原生组件。pipeline/CLI/public header胶合由root负责；没有替用户晋升默认。root已报告512×384 saved-candidate图像诊断25/25+PNG通过，但那份仍native_acceptance_eligible=false，不能把它改写为本原生组件的完整图像结果。32桶、1080-token、其他精度/fixture仍需对应证据后另行决策。


## 315-token / 2048桶真实原生桥接

组件提交`d9f3b4b`之后，复用已经复制的native runner（SHA `91ec9c70d04c0eaf14277076177941d64d4c875b4dc71fc25274265f8d19c452`）执行原`turbo512x384-s2048-portable`包的25块、CPU2、`--text-down-vector`，无trace、无诊断graph或中间tensor读取。`outputs/q2-native-text-vector-2048-v1/`保留独立worker/runner/runtime snapshots、完整模型/图SHA、315个输入ID及SHA、过程记录、结果和输出。身份进一步链接原runner来源的`q2-native-text-vector-v1/identity.json` SHA，避免将并行工作期间的其他源码变化误当该二进制来源。所有模型/图SHA逐个校验manifest及旧候选身份。

session44886 exit0，PID1684511，完整25/25块成功。最终FP32输出SHA `433ae46eaa1c8b831fa273f68b8345646d6ddfdb56695b4d3286b4964bca294a`，**逐位等于**此前独立diagnostic候选。对固定官方最后层SHA `2ef687ce362212b6dec95f953f7465d2131972b072ab7ed3e5b2ee39e420d01e` 的NRMSE为3.1361886512531567e-6，max为0.00091552734375，未调整门槛。

load10.833529384秒、compute168.156810604秒、外部wall180.104582490秒；10Hz采样峰值RSS1233328KiB，最低host可用13771968KiB。每次一块权重，RSS>2GiB或host可用<3GiB会停止，实际未触限；采样不保证捕获所有瞬时峰值。无GPU调用。此次证明生产text路径复现已测315-token候选，尚非PE→native text→DiT/VAE完整图像验收；默认仍Gemm，其他桶/精度覆盖仍pending。

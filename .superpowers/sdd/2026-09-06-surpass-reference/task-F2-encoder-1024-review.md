# F2 1024 encoder 独立复核

2026-09-06；审查作者提交d5d64c0及实际official-v2/native-v3证据。只读工具、fixture、param/bin和小输出，没有修改production、whitelist、数学、CMake，也没有执行模型/GPU。独立复算脚本与输出在 outputs/img2img-encoder-1024-review-v1。

## 结论

固定1024×1024候选的三边界数值证据成立，未发现该固定graph/weights/input上的剩余数值阻断。可以据此进入**显式固定1024 trusted入口**的下一实现/验证步骤：绑定本次完整param/bin/官方身份，保留已有512入口和旧包兼容，再独立验证生产包/API；不自动宣称生产入口已完成，不扩展任意尺寸、decoder/denoising/full img2img或正式质量/资源矩阵。

源码记录并非hermetic执行，主机available也不是连续guard；作者已经明确记录这些限制，本次只认可固定实际边界结果，不认可完整可重放环境或正式资源上界。生产shape校验仍明确排除1024，私有encode_vae_candidate_1024入口仅供证据probe。无数值默认被替换。

## Important发现及收敛

1. 初始reference_vae_encoder_large未执行独立wrapper，却固定写三个wrapper_bitwise_equal=true。这是可直接复现的错误证据声明。作者保留旧目录为 invalid-wrapper-metadata-v1，没有改其字节；重跑official-v2，改wrapper_comparison_status=not_run，并记录实际installed diffusers元数据。新旧三参考tensor字节相同。该项已关闭；不能重新引用旧fixture作为有效metadata。
2. 起初“完整source snapshot/host guard”范围不足：实际v2命令仍执行live tools路径，八份选定源码副本缺若干导入依赖；三份额外依赖只有postrun hash。worker无连续hostavailable监控。作者报告现在明确selected snapshot非hermetic、live-tools执行、postrun hash不足以证明运行期间不变、hostavailable只pre/post。该过度声明已关闭，原始证据限制仍保留，不冒充full source isolation或continuous host guard。

非阻断测试弱点已告知作者：test_image_encoder新shape负例只要求任意exception，同时输入坏graph和missing weights，故删除shape检查后也可能因加载失败而通过。需要具体shape-error断言才能证明加载前拒绝；当前production实现本身有正确显式shape检查。Python三项准备测试3/3通过。

## 实际数学/身份复核

- 有效官方fixture-v2 SHA88f2e8b7ad63a47fd993b069282dcb8bca9042cf56a470efa84237b711d9f7d9；official revision bc68c81e2a1730a394d5fc9fae70713dee940140。
- 本机实际diffusers direct_url指向7643c4826609c47755e3da0e5b768e8070468f49 archive；AutoencoderKLFlux2与DiagonalGaussianDistribution当前源码SHA分别7d9a976c…和8e6abad3…，匹配实际fixture和保存源码。load_encoder验证配置来源、encoder/quant/BN官方权重清单并strict装载；模式为eval FP32，后验mode不采样。
- RGB SHA08ea7276…，3,145,728 bytes HWC RGB。独立NumPy `(float32(v)-127.5)*float32(1/127.5)` 与保存in0逐位一致；mean按pixel_unshuffle2重排与packed逐位一致。
- 从官方小BN权重独立重算：eps1e-4与参考normalized最大差2.38418579e-7；错误eps1e-5差4.91142273e-5。graph实际BatchNorm1=0.0001、Reorg0=2/1=0、Crop取前32mean通道，与encoder合同一致；decoder inverse BN的1e-5明确属于另一方向，未混用。
- specialized-v2 param完整SHA d3207b56f558d65b9901ff73640b51ae2a0934143b43eaeab6cad45e275b9ceb。将两条conversion after逐字逆换为before，恢复全文SHA75d49399…，证明只有指定两行变化（含空白格式），不是仅检查局部行名。reshape_77变16384×512、reshape_78变128×128×512。全部bin SHA7fa2441a…与base相同。
- native冻结runner SHA368a2d2c1a963adf65c8506752932685faefda10bc351f5c1f119d3879e4c835及五份组件源码副本hash复核通过。其执行时input/param/bin绑定原v1路径；原目录后来被明确重命名，新v2同一输入/param/bin SHA一致，因此只建立字节等价桥，不把v3执行历史改写为直接使用v2 metadata。

## 三边界独立全量复算

严格集合mean/packed/normalized，形状分别[1,32,128,128]、[1,128,64,64]、[1,128,64,64]；actual/reference每项524288个FP32值，有限、完整文件大小和SHA核验。门槛保持NRMSE2e-5以及max_abs<=2e-4+2e-4*max_abs(reference)。

| 边界 | NRMSE | max_abs | max_abs门槛 | 结果 |
|---|---:|---:|---:|---|
| mean | 7.283859634807069e-7 | 6.9141387939453125e-6 | .0014835345268249514 | PASS |
| packed | 7.283859634807068e-7 | 6.9141387939453125e-6 | .0014835345268249514 | PASS |
| normalized | 7.33102113780981e-7 | 3.933906555175781e-6 | .0009088779926300049 | PASS |

与作者结果一致，分母3/3，没有missing/empty集合的vacuous pass。这里只测三个encoder边界，不能算decoder或完整img2img通过。

## 资源证据实际范围

官方v2退出0，26.12s，ru_maxrss3,547,740KiB，无swap；worker内部记录affinity4,6、memory.max17179869184、memory.swap.max0，可支持该scope生命周期硬限制。native-v3退出0，49.49s、ru_maxrss3,476,084KiB，无swap；配置8GiB/swap0/1800s，未从native scope内部保存effective cgroup值，报告已明示，不能当独立观测的硬边界证明。两侧hostavailable只有pre/post值，不推断全程最小值。ru_maxrss不是cgroup总内存，也不是并行后代RSSsum；没有GPU执行或设备内存测量。

## 测试弱点闭环

作者36c217c为两个关键shape负例增加精确错误消息断言，避免missing graph/weights的后续失败替代加载前shape拒绝。独立查看该diff并执行现有build-dev/ernie-image-encoder-contract成功；未配置或构建共享目录、未加载模型/GPU。该非阻断测试项也已关闭，源码/资源记录限制保持上述范围。

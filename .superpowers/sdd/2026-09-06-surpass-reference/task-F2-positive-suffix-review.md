# F2 官方 positive-strength suffix 独立审查

范围：提交 4edcfb3 的 tools/reference_img2img_positive.py、tests/test_img2img_reference.py，以及 outputs/f2-positive05-512x384-v1 的官方输入/suffix 证据。审查时这两个代码文件相对此提交无修改。仅静态和两物理核小 NumPy/stdlib 反例，没有 torch 导入、模型、GPU、构建或修改实现。

结论：无 Critical；两个可复现 Important 未关闭。它们是输入/输出认证链缺口，不代表这次已经保存的真实 suffix 数据错误。现有数据的轻量独立复核通过，详见下文。

## I1 — 受信 encoder fixture 未绑定实际边界和 start 混合（OPEN）

位置：tools/reference_img2img_positive.py:29-43、59-70。

代码只检查 fixture.json 的固定 whole-file SHA，然后独立验证 noise/start 与输入 contract 自报 SHA 相同。没有验证实际 RGB/encoder out0/out1/out2 与已认证 fixture 的对应 SHA，也没有从已认证 out2 与 noise 重算 sigma[4] 混合。request.sigma、sigmas_f32、encoder_fixture.normalized_sha256/encoder_bn_eps 不参加检查。后续把这些未认证边界复制并自签进 reference.json，scope 仍声称 official encoder/BN 和真实 noise 混合。

实际受信 fixture 保持完全不变，以下 CPU 反例均被 validate_inputs 接受：

1. start-4.f32 改为全零，并仅更新 input-contract 的 start SHA；接受。
2. out2.f32 改为同大小全零，不更新任何 fixture/contract；接受。
3. request.sigma=.123、encoder_bn_eps=1、sigmas_f32=[0]；接受。

这使同一个 reviewed encoder fixture 可为完全不同的实际 latent 背书。noise 可以是显式允许的不同已保存噪声，但 start 必须由已认证 encoded/noise/原八步 sigma 计算并逐字节核对。建议固定/校验实际 RGB 和全部 encoder 边界，再严格验证 BN、shape、类型、schedule、start 算式；不要仅增加可重新签名字段。

## I2 — complete 只转发 helper 标志，未检查 suffix 完整分母/执行合同（OPEN）

位置：tools/reference_img2img_positive.py:46-74，尤其57-58、64、67-71。

run 调用 reference 后只读取 complete is True 和 fixture 文件 SHA，没有检查返回值与落盘 fixture 一致，也没有检查 steps=8/start_step=4、恰好 prediction/step 4/5/6/7、六输入与三 final、shape/dtype/实际 SHA/PNG，或 initial 对 contract.start 的绑定、prompt 对请求的绑定。输出 schema 可对一个缺步骤/错误 prompt 的 fixture 宣称 complete=true。模型参数也未在此 wrapper 中绑定到固定 reviewed 512×384 config/package identity。

可复现的小型依赖 stub（明确不是实际模型执行）：让导入的 reference 返回并保存 `{complete:true,steps:2,start_step:0,outputs:[],final:{},prompt:'wrong'}`；用真实有效输入调用 run，输出 reference.json 仍 complete=true。该反例验证封装层无法拒绝上游回归/错误输出，不声称当前固定 helper 实际运行会产生该返回值。

建议为 suffix 添加独立 stdlib 合同，在写 complete 前验证精确 17 张量分母、4/5/6/7 文件身份、最终三边界和PNG、请求/prompt/initial/shape/精度与执行源绑定。8-step start4 必须单独标作四步 suffix，不能送入完整8步25项门槛。固定源码执行快照的身份应由独立执行记录或受审查 registry 认证；生成器自报 source_sha256 不是独立信任根。

## 实际历史数据的正证据与范围

独立读真实输入和官方输出，确认：

- encoder fixture whole-file SHA 是固定 `ee5ce5db3cb9912ff5e824bf062d1376bcf9b2e91e942d06924ecd187a93e1a9`，其中固化官方 revision、encoder/quant/BN 源和权重 SHA、mode 首32通道、pixel_unshuffle2、encoder BN eps1e-4/affine=false、decoder inverse eps1e-5。实际 input.rgb 和 out0/out1/out2 均匹配此受信 fixture。
- 实际 start-4.f32 逐字节等于 FP32 `.800000011920929*saved_noise + (1-.800000011920929)*out2`。实际噪声和起点 SHA 分别83226a3c…b6467bd和958c63fe…40e89d0；本次确实不是任意起点。
- suffix 的 steps=8/start_step=4，输出恰好4/5/6/7；原6输入+8预测/更新+3最终=17个张量的实际 SHA/byte size 全匹配，PNG SHA也匹配。reference.json嵌入suffix与落盘fixture完全一致，suffix SHA `378085a8be7c7b29df1400334c7ff9156d9deca4bab80879e5e6c23e94614989`。
- 静态核对 tools/validate_pipeline.py:62-64、74：先生成原8步schedule并set_begin_index(4)，再遍历原timesteps[4:]，未重建四步schedule；:97-100 inverse BN 明确eps1e-5。
- 实际 prompt是红苹果句子，9个有效文本token；padded text为2048，latent为[1,128,24,32]，decoded为[1,3,384,512]。官方 text 来自 real_reference；request.text_reduction=vector是native比较模式，不是官方参考实现使用native Vector reduction的证据。

没有重新散列全部模型权重、运行官方 encoder/DiT，或重新确认所有外部执行封存导入链。实际 fixture 记录 source_sha256=`53d6dde4c6a0542c0fa8dd1173886092b8cdebea4ad5124139fd6c1396236150` 和官方 pipeline SHA；本审查把它们视为记录字段，不以自签字段独立证明执行源。根持有外部worker/guard身份，应在最终证据包保留并独立绑定。request.device仍为pending_gpu是准备态记录；实际fixture环境记录DiT cuda、其余CPU，不把前者当实测设备。

## 重现

保存的独立小脚本：q2-review-scratch/f2_positive_repro.py。只在临时副本上制造负例，历史输入/实现未改。

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 taskset -c 0,2 .venv/bin/python .superpowers/sdd/2026-09-06-surpass-reference/q2-review-scratch/f2_positive_repro.py
```

实际输出：arbitrary_start ACCEPTED；wrong_encoder_bytes ACCEPTED；wrong_sigma_and_bn ACCEPTED；empty_wrong_suffix_complete True；actual_encoder_start_and_suffix 17 tensor hashes/sizes and PNG verified。

现有测试新增项只拒绝自签的假 encoder fixture，没有覆盖“真受信fixture+错误实际边界/混合”或suffix分母。此次完成后Q2两个GPU worker继续封存等待根明确放行。

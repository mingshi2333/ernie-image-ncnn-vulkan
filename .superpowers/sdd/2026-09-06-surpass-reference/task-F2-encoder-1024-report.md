# F2 1024×1024 encoder preparation

## Static graph and math contract

The reviewed 32×32 encoder template has exactly two spatial Reshape records. For a 1024×1024 RGB input,
the official encoder mean is `[1,32,128,128]`, and pixel-unshuffle 2 produces normalized packed latent
`[1,128,64,64]`. Therefore the only proposed graph substitutions are:

- `reshape_77`: `0=16 1=512` to `0=16384 1=512` (`128*128`, with channel width unchanged).
- `reshape_78`: `0=4 1=4 2=512` to `0=128 1=128 2=512`.

The new specialization helper rejects any other original Reshape content, any missing/extra Reshape, and any
shape other than 1024×1024. It does not change convolution, attention, GroupNorm, Crop, Reorg, weights or blob
names. This is a proposed shape-only graph contract; it is not yet admitted into schema-3 or runtime shape
whitelists.

The official reference producer continues to use posterior mode (first 32 mean channels), pixel-unshuffle 2,
encoder BatchNorm epsilon `1e-4` without affine parameters, and records decoder inverse-BN epsilon `1e-5`.
It authenticates the same pinned official encoder/quant/BN modules through `load_encoder`. The 512×384 producer,
fixture and trusted package entry remain unchanged.

The 1024 file is deliberately a thin reviewed configuration entry: deterministic input preparation, fixed dimensions,
resource guard and invocation of `reference_vae_encoder_large.run_reference`. All model loading, official source
authentication, normalization, posterior selection, packing, BN and tensor serialization remain in that shared
producer. Likewise, both 512×384 and proposed 1024×1024 records use
`specialize_vae_encoder.specialize_lines`; the 1024 wrapper only selects its explicit registered replacement.
No second implementation of encoder mathematics was introduced.

## Frozen development input

`outputs/img2img-encoder-preparation-1024x1024-v1` contains a new deterministic development-only RGB input;
it is not derived from or added to formal15/72. The HWC RGB file has 3,145,728 bytes and SHA-256
`08ea7276e09b38e7e1273eb5292dd56cab87d73c3a59032af34c8dae2f819153`. Its manifest fixes the dimensions,
layout, hash and resource plan before model execution.

## Resource estimate and guard

At 1024×1024, the encoder attention position count is 16,384. One dense FP32 `positions × positions` tensor is
1,073,741,824 bytes by itself. The 512×384 encoder mean is `64×48`, or 3,072 attention positions, because
the encoder downsamples each spatial axis by eight before attention. Thus 1024×1024 has 5.333… times as many
positions and about 28.444 times as many dense attention elements as 512×384. The earlier sub-3-GiB execution
cannot be safely extrapolated. Scores,
probabilities, activations, weights and framework workspaces can coexist.

The planned official run therefore uses two CPU threads, `MemoryMax=16G`, `MemorySwapMax=0`, a fresh output
directory and `/usr/bin/time -v`, only after root confirms sufficient host memory and no competing large job.
This 16-GiB value is a guard, not predicted consumption or a budget claim. A limit failure remains an honest
resource-incomplete result and must not trigger an unbounded retry.

## Bounded official and native execution

The first official run completed numerically but is retained under
`outputs/img2img-encoder-reference-1024x1024-invalid-wrapper-metadata-v1`: its producer claimed
`wrapper_bitwise_equal=[true,true,true]` without running such a comparison. It is not an eligible oracle.
The producer now records `wrapper_comparison_status=not_run` and the installed diffusers version, package path,
direct URL and actual implementation source hashes. The valid rerun is
`outputs/img2img-encoder-reference-1024x1024-v2`. Its effective scope record shows CPU 4,6,
`memory.max=17179869184`, `memory.swap.max=0`; it exited 0 in 26.12 s with peak RSS 3,547,740 KiB and zero swaps.
The actual outputs are finite FP32 with shapes `[1,32,128,128]`, `[1,128,64,64]`, and `[1,128,64,64]`.

The shape-only candidate is `outputs/img2img-encoder-1024x1024-specialized-v2`: param SHA-256
`d3207b56f558d65b9901ff73640b51ae2a0934143b43eaeab6cad45e275b9ceb`; its unchanged weight file remains
`7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce`. Production `encode_vae` still rejects
1024. A private fixed-size evidence entry rejects every non-1024 shape and was used by the frozen probe only.

The native v3 process used CPU 4,6, two threads, an 8-GiB systemd scope, swap limit zero and a 30-minute timeout.
It exited 0 in 49.49 s, peak RSS was 3,476,084 KiB, and `/usr/bin/time` reported zero swaps. Its runner and
component sources were copied before execution. The comparison at
`outputs/img2img-encoder-native-1024x1024-v1/result.json` binds the valid v2 official fixture and all actual
tensor hashes. Mean/packed/normalized NRMSE values are respectively `7.284e-7`, `7.284e-7`, and `7.331e-7`;
maximum absolute errors are `6.914e-6`, `6.914e-6`, and `3.934e-6`. All pass the unchanged FP32 gates
(`atol=2e-4`, `rtol=2e-4`, `nrmse=2e-5`). The v3 run did not capture effective cgroup files from inside the
scope, so its JSON states that limitation rather than presenting the configured limit as an observed value.

## Verification and pending work

`python -m unittest tests.test_vae_encoder_1024` passes 3/3. It checks deterministic input dimensions/type,
latent shapes, the 1-GiB single-matrix calculation, the exact two substitutions and fail-closed alternate shapes
or graphs.

No GPU, decoder, denoising, production package instance, formal input or full pipeline was run. Independent review
of the valid v2 source identity, exactly two graph changes, resource evidence and three-boundary comparison remains
required. Only after that review may a fixed 1024 trusted schema-3 instance be considered. Arbitrary encoder
dimensions remain unsupported.

The v2 command executed the live tools path. Its process directory contains selected producer and installed
implementation source copies, while `postrun-import-dependencies.sha256` records three further imported modules
after execution. It is not described as a complete hermetic source snapshot: `audit_port_weights.py` was already
modified in the preflight Git state, and post-run hashes alone cannot prove it stayed unchanged during the 26-second
run. The actual official implementation class files and weight/config identities are fixed in the fixture; independent
review must decide whether this source-recording limitation requires another hermetic metadata rerun. The host
`MemAvailable` >=3 GiB condition was checked immediately before launch and recorded again after exit, not monitored
continuously. The effective cgroup hard limit and swap prohibition applied for the full worker lifetime.

## Trusted fixed production registration

After independent review commits `773b00c` and `a6d205d` closed the candidate findings, schema-3 registers the
fixed source manifest `72bb195a...` with the reviewed 1024 param, unchanged encoder bin, source-instance BN files,
official fixture and conversion hashes. The existing 512 entry remains byte-for-byte present. Production
`encode_vae` now admits exactly 1024x1024 through the same component path and math used by 512x384; the temporary
evidence-only function was removed. Other shapes remain rejected.

`outputs/f2-production-img2img-package-1024-v1` was created by the normal schema-3 builder from the pinned
`models/turbo1024-s64-portable` instance and the reviewed specialized-v2 encoder. Python verification reports one
instance and 80 shared objects; manifest SHA-256 is
`2c1d0cdf39fe4cc94f7133d6dfac97a048123a8ebc8a4a5072e4d5e48368a5ad`. Generation quality remains pending.
The subsequent 1024 strength-zero production execution is intentionally reported separately after review.

## Real production strength-zero execution

Independent pre-run review found no blocker in `ec5f245`. The first prepared production supervisor had an
indentation error and exited before `Popen`; it remains in `outputs/f2-production-strength0-1024-v1-execution`
and is not execution evidence. The corrected, newly frozen run is
`outputs/f2-production-strength0-1024-v2-execution`. Its runner SHA-256 is
`ca2ed9dd12c9e3fc1a4d7b8e0869ef1db51ed20e75b4392801b00e769484c75e`; input PNG SHA-256 is
`1b0a823adc4cfc86befa8578d4a8369d72eff5cffd03bf201cdee45993d8c0b7`, and its decoded RGB is exactly the
reviewed development input SHA `08ea7276...`. The full source inventory was copied before launch.

The actual CLI command used the trusted schema-3 package, `--strength 0`, CPU FP32/direct VAE, two threads and
no prompt, PE or embeddings. The continuously polling supervisor observed the scope, `memory.max=8589934592`,
`memory.swap.max=0`, peak `memory.current=6083784704`, and minimum host MemAvailable `11252105216` bytes.
There were zero cgroup max/OOM events. It exited 0 in 151.47 seconds. Trace `final.f32` is byte-identical to
`encoder-normalized.f32`, so no denoising step was executed.

The official decoder continuation now shares one implementation for exact reviewed 512x384 and 1024x1024
fixture hashes; it does not admit other sizes or duplicate decoder mathematics. Its 1024 CPU run completed under
the same 8-GiB/no-swap/continuous-host guard in 51.47 seconds, peak `memory.current=4990722048`, with zero
OOM/max events. Production versus official NRMSE for mean, packed, normalized, final, unpacked and decoded is
respectively `7.284e-7`, `7.284e-7`, `7.331e-7`, `7.331e-7`, `7.299e-7`, and `2.307e-6`; maximum decoded
absolute error is `2.158e-5`. All six pass the unchanged FP32 gates. PNG max channel difference is 1 and mean
absolute difference is `0.000105858`. These results are bound in the production `result.json` and remain one
fixed development image, not formal15/72 acceptance.

The first decoder continuation (`img2img-reference-reconstruction-1024x1024-v1`) produced correct bytes but did
not record the installed decoder implementation identity; it is retained as source-incomplete. The v2 rerun records
the actual AutoencoderKLFlux2 source SHA, installed diffusers version/direct URL, VAE config, weight-manifest hashes
and a pre-run source inventory. Its reference JSON SHA is
`fc915a48ce5e2921d899ed7e8f33878f92686987866640e9e7d9181dbac00616`; v1/v2 unpacked, decoded and PNG bytes
are identical. V2 completed in 55.25 seconds with observed peak cgroup memory 4,876,550,144 bytes, minimum host
available memory 11,154,571,264 bytes, swap limit zero and no OOM/max event. The production result now binds v2;
the earlier result JSON is preserved as `result-source-incomplete-official-v1.json`.

## 固定 1024×1024 strength=0.5 准备与执行

在既有 `reference_img2img_positive.py` 中加入受信 profile registry，复用同一套输入认证、官方 text/DiT/decoder 数学和 17 张量 suffix 分母，没有复制第二套 1024 模型实现。registry 只允许已经独立审查的 512×384 和 1024×1024；1024 固定绑定 encoder fixture `88f2e8b7...`、schema-2 官方源 manifest `72bb195a...`、latent `[1,128,64,64]`、64-token text/DiT 图。任意其他尺寸 fail closed。

准备输入位于 `outputs/f2-positive05-1024x1024-v1/inputs`。它复用同一公开 deterministic 1024 PNG（SHA `1b0a823a...`，decoded RGB SHA `08ea7276...`）和受审查官方 encoder out2（SHA `29d24488...`），prompt 是精确 bytes `A red apple on a wooden table.\n`（SHA `a57aa1ee...`），PE off、Vector FP32。新 noise 由 torch 2.12.1 CPU `torch.randn`、seed 20260906 产生，little-endian FP32 `[1,128,64,64]`，SHA `9e11124f...`；原 8-step schedule 的 sigma[4] `0.800000011920929` 按 FP32 乘后加生成 start-4，SHA `f2f6cf88...`。输入合同 SHA `9fd1bfbb...`，生成后由正式 validator 重新读取、核对 PNG decoded RGB、所有 encoder 边界、schedule、finite/shape/hash 和 start 逐字节公式。

1024 固定图序列长度为 `64*64+64=4160`；旧 512×384/2048-token 图是 `32*24+2048=2816`，attention 元素比约 2.1823。旧有效 512 official suffix 峰 RSS 3,322,816 KiB、93.88 秒只能作为下界/参考，不能推导 1024 GPU 成功。`prep-identity.json` SHA `0527f8d8...` 保存了输入、准备源码和执行计划；状态明确为 `prepared_pending_independent_review_and_gpu_execution`。GPU 当前仍由 frozen 的 opt-in downGemm 候选独占，本轮未启动 official/native DiT，也未触碰 formal15/72。

小型验证：`.venv/bin/python -m unittest tests.test_img2img_reference -v`，9/9 PASS。新增测试固定 1024 profile/shape/source identity，拒绝未审查尺寸并验证错误 latent shape 在模型加载前失败。

## 固定 1024×1024 strength=0.5 实际执行

前置独立审查提交 `f04546f` 核对10个输入和10个准备源码、PNG decoded RGB、精确31-byte LF prompt、固定source manifest，并独立逐位重算524288个start元素；无阻断，只批准准备态。

两次冻结布局失败均原样保留且不算模型结果：`official-execution` 在2.62秒因snapshot层级令ROOT指错tokenizer而exit1；`official-execution-v2` 在4.18秒因repo-shape snapshot根缺`sources.lock.json`而exit1。两者均未进入DiT。v3仅补齐相同85个工具字节的repo目录形状、实际models绝对symlink及固定lock，identity SHA `a353965c...`。

有效official在`official-execution-v3`：exit0，wall181.194秒，10GiB cgroup实际可见，swap0，OOM/max events均0，peak memory.current 8,690,458,624 bytes，host available最低10,926,256,128 bytes。绝对步4/5/6/7全部执行；reference SHA `911bf45d...`，suffix fixture SHA `f62c9f9c...`，PNG SHA `d249bd95...`，validator确认17张量完整分母及精确token IDs末尾1626。

随后同一GPU串行运行冻结production runner `ca2ed9dd...`，证据在`native-execution`。命令显式使用同一PNG、LF prompt file、saved noise、strength0.5、8 steps、1024²、Vulkan FP32、Vector FP32 text、CPU direct VAE、threads2和新trace。exit0，wall494.307秒，peak memory.current 6,679,691,264 bytes，host available最低11,049,607,168 bytes，swap0/OOM0；四步及VAE/PNG完成。GPU随后明确释放给frozen。

新增`validate_img2img_positive_result.py`重新认证输入合同、official 17分母、两侧process/cgroup、prompt bytes/token IDs/RGB，并对encoder三边界、noise和17个suffix边界共21张量及PNG重算既定预注册FP32/conditioning/pixel gate。`result.json` SHA `94fdea75...`状态pass，comparison SHA `1ee35e56...`。最差张量是prediction-6 NRMSE `9.15499e-6`；final NRMSE `4.23103e-6`，decoded `7.70356e-6`；PNG max 1、MAE `0.000256856`，全部通过。该结果只覆盖一个固定公开development输入，不代表formal15/72或任意1024输入质量。

验证：`.venv/bin/python -m unittest tests.test_img2img_reference -v`，10/10 PASS；`.venv/bin/python tools/validate_img2img_positive_result.py outputs/f2-positive05-1024x1024-v1`，exit0/status pass。

独立末审指出首版post-audit未重新验证identity内容及完整资源字段。修复后会逐文件核对official 85源码+lock、installed runtime源码、schema2 manifest，以及native 249源码、runner和schema3 package manifest；process必须满足实际memory.max等于10GiB、swap0、host最低值不低于门槛、无failure且OOM为0。资源越界/failure小反例已加入。实际artifact在更严格validator下仍pass，11/11 unit PASS；没有重跑模型。

最终独立复核提交`c2a9ef6`重新计算完整21张量和PNG，并核对两侧执行源码、runner、runtime、package与guard。结论为该固定公开development样例PASS且无剩余阻断；不覆盖formal15/72、其他shape/strength、PE、低精度、性能或其他平台。

## 固定 1024×1024 strength=1 准备（未执行模型）

下一最小端点复用完全相同的公开图片、31-byte LF prompt、seed 20260906 saved noise、encoder fixture和受信package。输入合同SHA为`a269a01c...`；`steps=8`、`start_step=0`、`denoise_steps=8`、`sigma=1`，`start-0.f32`与`saved-noise.f32`均为2,097,152 bytes且SHA同为`9e11124f...`，并已逐字节确认相同。

工具仅把已有固定positive reference合同推广到两个显式允许值`.5/1`。strength1要求绝对步0..7；official suffix完整分母为25（6 conditioning输入、8组prediction/step、3最终输出），加encoder三边界与noise后端到端张量分母为29，另比较PNG。13/13 CPU小测试通过，覆盖bitwise noise端点和缺少任一八步时fail closed。

准备证据位于`outputs/f2-positive1-1024x1024-v1`，`prep-identity.json` SHA为`6b1f555f...`，状态明确为`prepared_pending_independent_review_and_gpu_execution`。未读取formal输入、未加载模型、未使用GPU；必须先经独立准备复核，随后才可按10GiB/swap0/连续3GiB host floor/1800秒guard排队执行official与native。

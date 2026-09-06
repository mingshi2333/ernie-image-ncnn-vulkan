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

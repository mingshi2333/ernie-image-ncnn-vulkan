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
1,073,741,824 bytes by itself. This is four times the positions and sixteen times the dense attention elements
of the 512×384 case (4,096 positions), so the earlier sub-3-GiB execution cannot be safely extrapolated. Scores,
probabilities, activations, weights and framework workspaces can coexist.

The planned official run therefore uses two CPU threads, `MemoryMax=16G`, `MemorySwapMax=0`, a fresh output
directory and `/usr/bin/time -v`, only after root confirms sufficient host memory and no competing large job.
This 16-GiB value is a guard, not predicted consumption or a budget claim. A limit failure remains an honest
resource-incomplete result and must not trigger an unbounded retry.

## Verification and pending work

`python -m unittest tests.test_vae_encoder_1024` passes 3/3. It checks deterministic input dimensions/type,
latent shapes, the 1-GiB single-matrix calculation, the exact two substitutions and fail-closed alternate shapes
or graphs.

No official encoder model, native encoder, GPU, formal input or full pipeline was run in this preparation slice.
The next gated sequence is: official CPU reference and source/hash freeze; specialize from the reviewed base;
native CPU FP32/direct three-boundary comparison; independent review; only then consider a fixed 1024 trusted
schema-3 instance. Arbitrary encoder dimensions remain unsupported.

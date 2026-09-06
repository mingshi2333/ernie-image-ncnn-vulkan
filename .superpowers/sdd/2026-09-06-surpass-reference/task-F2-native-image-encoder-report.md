# F2 native image encoder component report

## Component delivered

`encode_vae(ComponentFiles, RgbImage, ncnn::Option)` is a private numerical component. It accepts only the two independently reviewed shapes, 32x32 and 64x32, requires RGB byte count equality, CPU FP32 storage/arithmetic, direct convolution, and pack1 outputs. It converts HWC RGB bytes to CHW with the frozen `(float(value)-127.5f)*(1.f/127.5f)` order, loads the already authenticated in-memory graph plus immutable weight path, registers the reviewed custom layers, and extracts finite `mean`, `packed`, and BN-normalized boundaries at exact shapes.

The component deliberately does not parse packages, manifests, JSON, images, or CLI options. `ComponentFiles` authentication remains the caller's responsibility: schema3 uses `ModelPackage`, while this diagnostic uses the separately reviewed F2 verifier and pinned static graph hashes.

`ernie-img2img-runner` is a small diagnostic which reads exact RGB bytes, runs two-thread CPU FP32/direct convolution, and writes the three raw boundaries with checked close. It does not accept arbitrary resolution or perform GPU work.

## Tests

`ernie-image-encoder-contract` built successfully and `image_encoder_contract_cpu` passed. It verifies missing component data, unsupported 48x32 shape, incorrect RGB byte count, and Vulkan substitution are rejected before model execution.

Both the contract and diagnostic runner were built incrementally with two jobs. No full model or GPU computation was run.

## Actual 32x32 execution

The runner binary was copied to a fresh path (SHA-256 `783bf85cf3e5c0eebe42eadc80ec863feb1de244d9bc95cc98c11ed2620ec522`) and run inside a `MemoryMax=2147483648`, no-swap user scope against the reviewed graph `75d493...`, weights `7fa244...`, and the frozen `input.rgb` from `img2img-encoder-32x32-v1`.

The real native boundary hashes were:

- mean: `079c40197c37db1ed17394e22db6fef8d4c0d2e58b5ab96c073854ebc1f89a82`
- packed: `ffab2e18b69c40f4ffe6f74b541bd064a55957ddc3ff9f7a1d88bb12c0460996`
- normalized: `67358bf98b55edc778181b3c6210e96264a93e40bd5ef94f36b48a56b5ded05d`

Against the final official 32x32 reconstruction oracle:

| boundary | max abs | mean abs | NRMSE | cosine |
| --- | ---: | ---: | ---: | ---: |
| mean | 3.33786e-6 | 7.12902e-7 | 6.09489e-7 | 0.999999999999816 |
| packed | 3.33786e-6 | 7.12902e-7 | 6.09489e-7 | 0.999999999999816 |
| normalized | 1.90735e-6 | 4.03386e-7 | 6.11701e-7 | 0.999999999999815 |

All are far inside the pinned FP32 gates `atol=2e-4`, `rtol=2e-4`, `nrmse=2e-5`. They are not bitwise identical to the earlier runner snapshot, so this report records numerical gates rather than inventing a bitwise claim.

## Completed strength-zero reconstruction

The reviewed decoder template `models/vae-8x8-v1/head.ncnn.param` was rehashed as `6ecb591c473c56f3d9e923c845be2125e6e42b33ee2cd8d4d1783db68e32c4c5`. A new official reference-only 4x4 latent fixture was produced under the same 2 GiB/no-swap scope. `specialize_vae.py` then changed exactly the two reviewed absolute-shape parameters: `reshape_99` flatten `64 -> 16`, and `reshape_100` spatial `8,8 -> 4,4`. The resulting graph SHA-256 is `c0186972d563fec0bd189eaff72f69a6ff3726720a3b5730b0bfb7e4aad57394`; its model manifest is `482a97d0a475bd3188b0613fc332cfa86e300e32dd42cbb4b0e35c3485773981`, and the independent official fixture is `915cf88c10a3cd8a9179edfba02f3f279232ed48a5d93f3a681d53206118d8b5`.

The rebuilt runner (SHA-256 `47f309cfe73b649da9ae7c980ab86f64be8806c39b7ddaa31fc5aa888b5f905f`) executed the complete small native chain under a 2 GiB/no-swap scope:

`RGB bytes -> encode_vae -> make_img2img_start(strength=0) -> unpack_for_vae(eps=1e-5) -> decode_vae(direct)`

The start hash equals the normalized encoder hash, as required at strength zero. Against the final official reconstruction oracle, the additional boundaries were:

| boundary | max abs | mean abs | NRMSE | cosine |
| --- | ---: | ---: | ---: | ---: |
| start | 1.90735e-6 | 4.03386e-7 | 6.11701e-7 | 0.999999999999815 |
| unpacked | 3.33786e-6 | 7.15074e-7 | 6.11717e-7 | 0.999999999999815 |
| decoded | 3.57628e-6 | 6.48047e-7 | 1.59423e-6 | 0.999999999998737 |

The native decoded PNG is pixel-identical to the official PNG: max/mean pixel error `0/0`, with shared SHA-256 `af9609799e485e39cd8f5eda50dd8c4684c5bfaabca0e8d595a9ea950a95b2f9`.

The copied runner, six native raw boundaries, reconstructed PNG, and exact encoder/runner/specializer source snapshots are frozen under `outputs/img2img-native-reconstruction-32-v1`; its inventory manifest SHA-256 is `00aacbf726c2165f7bdb239aa336bc562a020eac14a81e05ed05326eb605fa3b`.

This completes the requested small strength-zero native reconstruction component loop. It does not establish schema3 production integration, positive-strength DiT behavior, arbitrary-shape decoder specialization, or 15-case completion.

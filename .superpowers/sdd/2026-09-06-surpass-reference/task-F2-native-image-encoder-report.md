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

## Incomplete boundary

The native strength-zero `make_img2img_start -> unpack_for_vae -> decode_vae` continuation is not yet executed in this slice because no reviewed 32x32 decoder param was present in the available outputs. The repository only exposed the existing 512x384 VAE param. A shape-edited decoder graph must be independently pinned before it can be used as evidence. Therefore this commit establishes the real native encoder component and its three official boundary comparisons, but does not claim the requested native reconstruction loop, schema3 production integration, or 15-case completion.

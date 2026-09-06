# F2 img2img start and denoiser suffix independent review

## Scope

Reviewed commit `400b460` against P3 Task F2, the frozen B1 protocol, and the pinned reference source. No production code was changed and no model or GPU process was run.

## Verdict

No Critical or Important finding in this bounded slice. It correctly implements the specified start-latent and denoiser-suffix contract, while its report accurately leaves encoder parity, pipeline integration, Vulkan suffix parity, and the 15 image/strength runs open.

## Verified behavior

- `make_img2img_start` rejects nonfinite/out-of-range strength, invalid thread counts, malformed/non-descending schedules, invalid encoded latents, and—for positive strength—missing, nonfinite, or mismatched saved noise (`src/img2img.cpp:12-34`). `finite_latent` supplies the full pack1 FP32 `[128,H,W]` contract for both inputs.
- Strength zero clones encoded storage, sets `start_step=8`, `denoise_steps=0`, `sigma=0`, and never reads noise (`src/img2img.cpp:25-31`). Strength one clones noise at sigma 1 (`src/img2img.cpp:35-43`). The intermediate operation order is two FP32 multiplies followed by an add (`src/img2img.cpp:47-57`), compiled under the existing no-contraction property for `img2img.cpp`.
- For eight steps, independent FP32 recomputation produced:

```text
strength  denoise_steps  start_step  sigma[start]
0         0              8           0
.25       2              6           0.5714285969734192
.5        4              4           0.800000011920929
.75       6              2           0.9230769276618958
1         8              0           1
```

This matches the frozen positive rule `min(steps,max(1,floor(steps*strength+0.5)))` and the pinned peer implementation at `outputs/reference-port-v1/source/src/ernie_image_pipeline.cpp:1790-1803`. The `.3125` half-step test also distinguishes round-half-up from bankers rounding.
- Both denoiser overloads build the original complete Turbo schedule and index it with absolute `i=start_step..steps-1`; observer indices, timestep, and delta therefore retain original-step identity (`src/denoiser.cpp:49-73`, `89-120`). Stats are cleared before the loop and contain only executed suffix entries. `start_step==steps` returns without loading a DiT graph, invoking an observer, or retaining stale stats. Invalid start values are rejected.
- The zero-step return may share ncnn storage with its supplied initial latent, but `make_img2img_start` already returns independent storage and denoising treats its input as read-only. This is not a correctness finding under the current contract.

## Test evidence

```text
ctest --test-dir build-dev --output-on-failure -R '^img2img_contract_cpu$'
1/1 passed (0.05 s)
```

The focused C++ test covers the five required strengths, exact endpoint bytes and non-aliasing at the start-construction boundary, half-step rounding, smallest positive strength, malformed input rejection, and the no-model/no-observer zero-step path.

## Open scope

This slice does not establish complete img2img behavior: no encoder/decoder composition, actual image, full denoising suffix, Vulkan parity, or 15-case quality result was exercised. Those remain required before F2 acceptance.

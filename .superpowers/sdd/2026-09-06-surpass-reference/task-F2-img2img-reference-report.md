# Task F2 official-module img2img reference report

## Delivered scope

- `tools/reference_img2img.py` defines the frozen FP32 strength contract, the native-compatible eight-step turbo sigma construction, and the exact `sigma * noise + (1 - sigma) * encoded` operation order.
- Strength uses FP32 round-half-up. For eight steps, strengths `0/.25/.5/.75/1` select `0/2/4/6/8` denoise steps and start indices `8/6/4/2/0`. Any positive representable strength selects at least one step.
- The executable path is deliberately limited to strength-zero reconstruction. It loads the pinned official encoder, quantizer, encoder BN, inverse BN, and decoder; it does not load text or DiT. A positive-strength CLI request fails explicitly until the full DiT oracle is integrated.
- `tools/validate_pipeline.reference()` has a compatible tail `start_step=0`. It validates the suffix, sets the official scheduler begin index, and executes only `timesteps[start_step:]` while retaining the complete schedule.
- The reference manifest binds the input RGB bytes and normalization, saved FP32 noise and seed, strength/steps/start index, posterior mode, pixel-unshuffle packing, encoder BN epsilon `1e-4`, decoder inverse-BN epsilon `1e-5`, official component hashes, installed source hashes, every saved tensor, and decoded PNG.

## Tests and negative checks

Command:

```text
python3 -m unittest tests.test_img2img_reference
```

Result: `Ran 4 tests ... OK`. These tests cover all five requested strengths, the `.3125` half case, the smallest positive FP32 strength, exact sigma bits, endpoint identities, intermediate FP32 operation order, independent zero-strength storage, and fail-closed invalid strength/dtype/nonfinite/bool inputs.

Command:

```text
python3 -m py_compile tools/reference_img2img.py tools/validate_pipeline.py tests/test_img2img_reference.py
```

Result: success.

An isolated call with `start_step=9, steps=8` failed before package/model access with `ValueError: Reference start step must be in [0,steps]`.

The repaired encoder validator from `b2179b1` was independently rerun with `.venv/bin/python -m unittest tests.test_img2img_encoder`: all 12 tests passed. Replaying the earlier resealed-source tamper now fails with `Encoder exporter/schema identity mismatch`, so the candidate used by this reference is no longer accepted merely because its own manifest was rewritten.

## Actual bounded official reconstruction

Final-code run:

```text
.venv/bin/python tools/reference_img2img.py --output /home/mingshi/Project/AI/ernie-image-ncnn-vulkan/outputs/img2img-official-reconstruction-32-v3 --width 32 --height 32 --steps 8 --strength 0 --seed 42
```

Result: passed under an observed user cgroup with `MemoryMax=2147483648`, no swap, no OOM event, peak cgroup memory `1548247040`, peak process-group RSS `1190346752`, and wall time `21.7952s`. The recorded tool SHA-256 equals the final file SHA-256: `976b63b20c0d0e2513de736bc96e3e0673fc0f30712c949fcb752d6d60baa7d0`.

The reference manifest SHA-256 is `e035cd8bce9ac7dda0c36f94ed228849cc7946768c46556d219e0dfbfefd251c`; the decoded PNG SHA-256 is `af9609799e485e39cd8f5eda50dd8c4684c5bfaabca0e8d595a9ea950a95b2f9`. All eight small tensor files were independently rehashed against the manifest. `start.f32` equals `encoded.f32` at strength zero, while saved noise has a separate identity. The decoded tensor is `[1,3,32,32]`.

An earlier v1 execution completed the numerical work but failed while serializing a NumPy scalar. It is retained as failed evidence and is not cited as a successful run. v2 succeeded before the final boolean-input guard; v3 is the evidence bound to the final tool.

## Explicitly incomplete

- Positive-strength official DiT suffix execution has not run. The pure start-latent and schedule-suffix interfaces are present, but no quality or parity claim is made from them alone.
- Strength one has the exact saved-noise start identity in the unit contract, but its full text/DiT trajectory has not been executed.
- No complete 12/15-case img2img acceptance batch, native comparison, GPU run, or full-resolution run was performed.
- This artifact establishes an official-module encoder/decoder reconstruction oracle. It is not evidence that Diffusers publishes or implements an ERNIE img2img pipeline.

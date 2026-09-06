# P3 F3 device diagnostics and execution-control report

## Result

Implemented the bounded F3 device/diagnostic continuation without running a model. The public stdlib-only API now exposes `diagnose()`, which initializes ncnn's compiled Vulkan runtime, copies the actual visible device records and precision-storage capabilities, and optionally parses the small `model.cfg`. It does not call package verification, traverse a weight manifest, or load weights. The CLI exposes this as `--diagnose`, which accepts only optional `--model DIR`; generation options cannot be mixed into diagnostic mode.

`GenerationRequest::threads` is now used by PE, text encoding, latent unpack, and the existing CPU option. The PE tail parameter defaults to 4 so existing callers remain source-compatible. `gpu_index` is validated against the initialized runtime before model metadata, PE, text, or image weights are read, and the same requested index is passed to DiT and Vulkan VAE decode. The former explicit-index/Vulkan-VAE refusal was removed because VAE now calls `set_vulkan_device` with that selected index. The VAE tail parameter defaults to -1 for existing callers.

The diagnostic API labels the current six-field static `model.cfg` contract as schema 1 and reports packed width/height, text bucket, DiT text tokens, and the required 25/36 layer counts. It does not claim schema-3 dynamic graph support or weight-package validity.

## Verification

Focused configure/build, with no model execution:

```text
cmake -S . -B build-dev -DERNIE_BUILD_GENERATOR=ON
cmake --build build-dev --target ernie-image ernie-pipeline-api-contract ernie-request-validation-contract -j2
exit 0
```

Actual local ncnn device discovery through the resulting CLI:

```text
./build-dev/ernie-image --diagnose
vulkan_compiled=true
gpu_count=3
default_gpu_index=0
gpu[0].name=NVIDIA GeForce RTX 4060 Laptop GPU
gpu[0].fp16_storage=true
gpu[0].bf16_storage=true
gpu[1].name=llvmpipe (LLVM 22.1.8, 256 bits)
gpu[1].fp16_storage=true
gpu[1].bf16_storage=false
gpu[2].name=llvmpipe (LLVM 17.0.6, 256 bits)
gpu[2].fp16_storage=true
gpu[2].bf16_storage=false
```

Contract and CLI tests:

```text
ctest --test-dir build-dev --output-on-failure -R '^(pipeline_api_contract|request_validation_contract)$'
2/2 passed

ERNIE_TEST_RUNNER=$PWD/build-dev/ernie-image python3 -m unittest tests.test_cli
Ran 24 tests in 1.871s
OK
```

The API test includes only the public header and checks copied device records/default selection. Request validation uses the discovered device count and proves an unavailable explicit VAE index is rejected before the absent model can load. CLI tests prove model-free discovery, bounded `model.cfg` parsing without weights, and rejection of mixed diagnostic/generation options.

## Remaining integration and runtime evidence

No complete PE/text/DiT/VAE model was run. Therefore end-to-end evidence that a non-default physical device completes generation, and evidence that a requested thread count changes all full-model executor pools, remain pending root-controlled model runs. The code paths are explicitly wired and the small runtime selection/validation contract is exercised.

F2 image encoding and F1 dynamic shape/schema-3 integration remain separate work. This change does not make img2img available and does not claim F3 as a whole complete.

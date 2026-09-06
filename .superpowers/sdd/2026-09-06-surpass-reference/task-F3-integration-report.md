# P3 Task F3 CLI/API integration report

## Scope and result

This change connects the previously reviewed `read_image`/`write_image` codec component to the generator's output path and adds the public request and CLI configuration surface required for later dynamic/img2img work. It does not claim full F3 completion. No model or GPU process was run.

The CLI now selects PNG, JPEG/JPG, BMP, or TGA from the output extension and rejects unknown extensions before model loading. Existing-output rejection remains unchanged, and the compatibility `write_png` API remains available in the codec component.

`GenerationRequest` now carries `threads`, `gpu_index`, `text_device`, optional RGB input, and `strength`. New members are appended after all existing request members so older positional aggregate initializers keep their meaning. CPU stage options receive the requested thread count. An explicit GPU index is used by the DiT Vulkan path and checked against the devices visible after ncnn initializes Vulkan. Text encoding remains CPU-only and rejects any other requested value.

The CLI recognizes `--input`, `--strength`, `--resize`, and `--background`, validates their syntax and dependency, then rejects every otherwise valid `--input` request before model loading with an explicit F2-unavailable error. The public API independently validates image dimensions, checked RGB byte count, optional explicit-resolution agreement, and finite strength before returning the same unsupported result. No input image is accepted and ignored.

The Vulkan VAE currently obtains ncnn's default device inside its existing component. Therefore a request combining an explicit GPU index with a Vulkan VAE is rejected before loading a model. Supporting that combination requires the later VAE/device plumbing rather than silently using a different device.

## CPU-only evidence

Configuration and focused build:

```text
cmake -S . -B build-dev
cmake --build build-dev --target ernie-image ernie-pipeline-api-contract ernie-request-validation-contract ernie-image-io-contract -j2
Result: all four targets built successfully.
```

Focused native contracts:

```text
ctest --test-dir build-dev --output-on-failure -R '^(pipeline_api_contract|request_validation_contract|image_io_contract)$'
Result: 3/3 passed.
```

CLI contract:

```text
ERNIE_TEST_RUNNER=$PWD/build-dev/ernie-image python3 -m unittest tests.test_cli -v
Result: 21/21 passed.
```

After moving the new public members to the end of the request structure, the pipeline API target was rebuilt and executed again successfully.

The request-validation contract covers thread bounds, GPU range/device compatibility, CPU-only text, finite strength, RGB byte length, input/request dimension mismatch, valid img2img fail-closed behavior, and the explicit-GPU/Vulkan-VAE limitation. The API contract also compiles an older positional aggregate initializer. Image format round trips remain covered by the codec contract.

## Remaining work

- F2 must implement input preprocessing, resize policies, background composition into actual RGB input, VAE encoding/noise injection, and strength endpoint semantics. Until then all img2img requests remain unsupported.
- The dynamic shape plan/package work from F1 must be integrated before zero dimensions or resized input can select arbitrary valid runtime shapes.
- Explicit GPU selection has not yet been passed through the Vulkan VAE component; that combination is fail-closed.
- A real 64x64 text-to-image generation and GPU/device-index execution remain for the root-controlled serial runtime check. This change provides compile and CPU request/codec evidence only.

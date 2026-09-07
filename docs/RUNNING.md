# Run the native Turbo generator

The first supported configuration is Linux, ERNIE-Image-Turbo, batch one,
8 Euler steps and CFG 1, with optional CPU prompt enhancement. Model dimensions and
maximum prompt length are properties of the selected static package.

Build with a C++17 compiler, CMake 3.19+, Cargo, libpng development headers,
and Vulkan development headers/driver when using a GPU:

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DERNIE_BUILD_TOKENIZER=ON -DERNIE_BUILD_GENERATOR=ON \
  -DNCNN_SYSTEM_GLSLANG=OFF -DNCNN_INT8=OFF -DNCNN_WEIGHT_QUANT=OFF
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/outputs/install"
```

Set `-DERNIE_ENABLE_VULKAN=OFF` for a CPU build. The ncnn revision is checked
against `sources.lock.json`, and Cargo uses its checked-in lockfile. Inference
uses C++ and the statically linked Rust tokenizer; Python and a network
connection are not required at runtime. A source-built executable still needs
compatible system libraries (libpng, the compiler's OpenMP/C++ runtime, and
the selected shader-toolchain libraries when dynamically linked). A model
package can be moved between compatible builds; this is not a universal Linux
binary distribution.

The root CMake file delegates targets to their owning directories. A CLI-only
configuration can add `-DBUILD_TESTING=OFF -DERNIE_BUILD_PROBES=OFF`. Conversion
and diagnostic runners still use the documented `build/ernie-*-runner` paths
when enabled. Applications can link `ernie::pipeline` and include
`ernie/pipeline.h`; generation returns RGB pixels and uses a progress callback,
without requiring command-line parsing or libpng. For a separate C++ application,
configure the producer with `-DERNIE_INSTALL_SDK=ON` and install it. This installs
the public header, pipeline archives, native tokenizer and the same pinned ncnn
used by the executable. Vulkan SDK builds use `NCNN_SYSTEM_GLSLANG=OFF` so the
shader compiler archives move with the prefix. Allocation-instrumented builds
remain separate diagnostic builds.

The external application's CMake file only needs:

```cmake
find_package(Ernie 0.1.0 EXACT CONFIG REQUIRED)
add_executable(my-app main.cpp)
target_link_libraries(my-app PRIVATE ernie::pipeline)
```

Configure that application with `-DCMAKE_PREFIX_PATH=/path/to/installation`.
It needs a compatible C++/OpenMP toolchain and the system libraries described
above; it does not need this checkout, Cargo, Python, PNG headers or ncnn headers
in its own source. The installed configuration always imports the bundled ncnn,
so load Ernie before a separate ncnn package. The public interface is a source
API; version 0.1.0 does not promise a stable binary ABI across toolchains.

Linux presets additionally include SDK installation and model-free tests
(CMake 3.21+ and Ninja): `cmake --preset linux-cpu` or
`cmake --preset linux-vulkan`, followed by the same `cmake --build --preset` and
`ctest --preset` name. Manual configuration above still supports CMake 3.19.
The installation consumer test moves the prefix into a path with spaces and
Chinese characters, builds a separate application, and checks package rejection.
The explicit Linux evidence command `python3 tests/test_install.py --build BUILD
--output NEW_DIRECTORY --isolate-source` additionally uses bubblewrap to hide the
checkout/build paths and disable networking. Its output must be outside the
checkout. These small installation checks do not execute ERNIE model inference
or establish Windows/macOS support. See `docs/CODE-STRUCTURE.md` for ownership.

Convert and assemble the official components using `docs/REPRODUCE-PIPELINE.md`
in the source repository. To turn an existing local symlink package into a portable model:

```sh
python tools/package_model.py --model models/pipeline1024-residual-v1 \
  --output models/turbo1024-portable
build/ernie-image --model models/turbo1024-portable --verify-model
build/ernie-image --model models/turbo1024-portable \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --seed 42 --steps 8 --output outputs/apple.png
```

The packaging script uses only the Python standard library. It validates source
manifests, copies the 136 files actually used in inference, and writes a flat
inventory of SHA256 checksums and byte lengths. No conversion fixtures,
intermediate graphs, or external symlinks are required by the portable package.
Use a new output directory. `--link` creates a development package that still
depends on the source files.

The executable verifies every inference file before loading weights, including
the tokenizer, final text/DiT blocks, BN statistics, and VAE. `--verify-model`
performs this check without loading the model or using a GPU. A full checksum
pass reads all model files and contributes to startup time. The manifest detects
missing and corrupted bytes; it is not a signature establishing who supplied a
package. Old local schema-1 packages remain supported.

The CPU VAE defaults to direct convolution, with FP32 activations and FP64
GroupNorm reductions. `--vae-convolution sgemm` retains the older, larger
workspace path for controlled comparisons. `--vae-device vulkan` is experimental
and is not the validated CPU VAE path. GPU DiT precision is FP16 by default;
`--precision fp32` retains the FP32 path. CPU inference requires
`--device cpu --precision fp32`.

`--precision bf16` enables experimental Vulkan BF16 storage after checking
device support. The 1080-token, 512x384 fixture completes all eight steps but
fails the fixed quality gates in BF16 and FP32. Its PNG MAE/max error is
10.3763/255 in BF16 and 0.08213/17 in FP32, on a 0..255 scale. Do not infer
quality acceptance from finite activations or BF16's exponent range.

## Prompt files and static variants

Use exactly one of `--prompt TEXT` or `--prompt-file UTF8.txt`. Files may have a
UTF-8 BOM; every other character, space and CRLF is preserved. Invalid UTF-8,
NUL bytes and files larger than 1 MiB are rejected before loading weights.

The verified text export capacities are 32, 64 and 2048 tokens, including BOS.
The local portable package `models/turbo512x384-s2048-portable` supports
512x384 output and all five prompts from the comparison repository by capacity.
One 1080-token example has been exercised through the complete native image
path. This does not imply that all five images have been validated.

```sh
build/ernie-image --model models/turbo512x384-s2048-portable \
  --prompt-file prompt.txt --width 512 --height 384 \
  --precision fp32 --output outputs/long-prompt-new.png
```

Both explicit dimensions must match the selected static package. To prepare
another size or text capacity from the baseline components:

```sh
.venv/bin/python tools/prepare_variant.py --width 512 --height 384 \
  --text-tokens 2048 --work outputs/my-variant-build \
  --output models/my-variant
```

The builder independently exports the new text/DiT shapes and heads, checks
complete graph hashes before weight reuse, and validates the target-resolution
VAE against an independently executed official fixture. Its current bounds are
16..1024 pixels per axis, multiples of 16, text capacity 1..2048, and at most
6144 combined tokens. A completed component build is followed by a separate
image-quality run. Use `package_model.py` without `--link` to materialize a
portable copy of the resulting development package.

An additional experimental fixed **1376x768/s64** package can be created from
the pinned `turbo1024-s64-portable` source. This path reuses the original weights
and changes only the audited spatial/token fields in all 64 component graphs.
The complete source manifest must match the reviewed 1024x1024/s64 package.

```sh
python tools/package_model.py --model models/turbo1024-s64-portable \
  --fixed-1376x768 --output models/turbo1376x768-s64-portable
build/ernie-image --model models/turbo1376x768-s64-portable --verify-model
build/ernie-image --model models/turbo1376x768-s64-portable \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --width 1376 --height 768 --steps 8 --precision fp32 --text-down-vector \
  --seed 42 --output outputs/apple-1376.png
```

The output uses the existing fixed-package protocol and contains 136 runtime
files. Add `--link` during packaging to share local source weights instead of
copying them; that development package requires the source to remain present.
The text capacity is 64 tokens including BOS, giving 4192 combined tokens.
This command produces a separate static package. A shared package can instead
instantiate this spatial target at runtime from the same 1024 source, as described
below. An encoder is not available at this shape. Broader prompt-quality
acceptance remains separate from the fixed development case.

The complete 1376x768 apple fixture has executed all eight native steps and CPU
decoding, but remains **quality_gate_failed**: 23/25 tensor checks pass;
prediction-7 and decoded fail their fixed maximum-error limits. PNG MAE is
0.005041/255 and max error is 3 versus the original limit of 2. The portable
package passes Python/native verification and matches all 136 runtime files of
that actual run. See [the full evidence](../artifacts/2026-09-07/fixed1376-native-pipeline/README.md).

## Shared weights and runtime dimensions

The schema-3 runtime now accepts an experimental bounded shape contract: each
axis is a multiple of 16 in [16,2048], with at most 2097152 pixels. Complete
image comparisons across the entire range are still pending. The installed
512x512 native FP32 example completes with the source hidden and networking
disabled, passing all 25 tensor checks and PNG max difference 1/255. The 64x64
example also passes the numerical comparison, but both official and native
images show gray textures. See the [actual image and SDK checks](../artifacts/2026-09-07/runtime-images-and-sdk/README.md).
Graph instantiation and
one maximum-length block are narrower evidence; see the
[current report](../artifacts/2026-09-07/runtime-range-and-buckets/README.md).

Build the three-bucket package once from the authenticated portable sources:

```sh
python tools/package_dynamic_model.py --schema3 \
  --source models/turbo512x384-s2048-portable \
  --source models/turbo1024-s64-portable \
  --source models/portable-turbo1024-s32-v1 --output models/turbo-shared-v2
build/ernie-image --model models/turbo-shared-v2 \
  --prompt 'A red apple on a wooden table, soft daylight, realistic photo.' \
  --width 512 --height 512 --precision fp32 --text-down-vector \
  --steps 8 --seed 20260905 --output outputs/shared-512.png
```

This command uses the native seeded noise generator. The recorded official/native
comparison instead shares exact saved FP32 noise bytes. In this development
checkout, adding `--latent outputs/runtime512-official-v2/initial.f32` selects
those bytes; an identical integer seed across frameworks is not sufficient.

Generation uses the native executable and this package. It never converts a
new model or writes temporary parameter files when the dimensions change.
The new bundle has 88 unique objects, about 21.67 GiB, including independently
exported 32/64/2048 text templates and one shared set of weights. Existing
two-source bundles continue to work with their available 64/2048 buckets.

The actual prompt is tokenized after optional PE. The smallest available bucket
that can hold all IDs, including BOS, is selected without rereading the weight
store. The 32-token source retains 64 DiT text slots. Valid length, text-encoder
bucket and DiT padding are distinct; right padding never becomes valid text.
Short prompts in existing multi-source shared bundles may therefore select a
smaller text template than their original geometry-selected default. Static
schema-1/2 packages preserve their fixed configuration.

For a combined image/text sequence above 6144 tokens, Vulkan keeps block weights
in system memory. This policy completed the 10240-token block experiment within
the existing resource guard. It is not a guarantee for every full image, driver,
or concurrent desktop workload. A trace records the actual selected shape,
text bucket, DiT padding and weight placement in `shape.txt`.

The earlier 1376x768/s64 full pipeline produced the same 25 tensors and PNG as
its fixed package. Its historical official comparison remains 23/25 numerical
checks and PNG maximum difference 3 versus the original cutoff 2; see the
[unaltered execution record](../artifacts/2026-09-07/runtime-shape1376/README.md).
These project-selected numerical tolerances remain uncalibrated for broad
perceptual quality; see [numerical diagnostics](NUMERICAL-DIAGNOSTICS.md).

Image-to-image encoder availability remains independent of decoder dimensions.
Only an encoder reviewed for the requested geometry is exposed; selecting a new
text bucket cannot make an encoder for another size available. The three-source
bundle above contains no encoder. Existing reviewed single-source encoder bundles
retain their original image-to-image path.

## Optional native prompt enhancement

PE is a separate actual Ministral3 model: 26 layers, final norm and LM head.
It does not replace the image encoder's 25-layer Mistral path. Prepare it in
the reference environment after initializing the baseline tools:

```sh
.venv/bin/python tools/fetch_pe_components.py
.venv/bin/python tools/export_pe_block.py --output models/my-pe-block00
.venv/bin/python tools/validate_pe_block.py --model models/my-pe-block00 \
  --output outputs/my-pe-cache-check
.venv/bin/python tools/build_pe_model.py --template models/my-pe-block00 \
  --validation outputs/my-pe-cache-check --output models/my-pe
build/ernie-image --pe-model models/my-pe --verify-model
```

The existing local complete package is `models/pe-cpu-v1`. It contains 60 runtime
files and no symlinks. Both Python and native checks cover its full inventory,
file sizes, hashes, revisions, cache configuration and pinned chat template.

```sh
build/ernie-image --model models/turbo512x384-s2048-portable \
  --pe-model models/pe-cpu-v1 --pe-greedy \
  --prompt 'A red apple on a wooden table.' --precision fp32 \
  --output outputs/pe-apple-new.png --trace-dir outputs/pe-apple-new-trace
```

Without `--pe-greedy`, sampling uses temperature 0.6, top-p 0.95 and PE seed 42.
Override these with `--pe-temperature`, `--pe-top-p`, and `--pe-seed`.
`--pe-max-tokens` is 2048 by default; a smaller value may end the description
before EOS, which the CLI reports explicitly. The PE chat input is capped at
2048 tokens without truncation, and its native cache at 4096 total positions.
This remains below the official model's position-dependent query-scaling
boundary at 16384. Prefill consumes exact tokens sequentially; batched prefill
and GPU PE are not implemented.

The complete greedy apple fixture reaches EOS at token 315 and passes every
logit comparison against official CPU FP32 `generate`; token IDs and final text
match exactly. Native run peak RSS was about 14.71 GiB. The full PE model is
released before image text encoding begins. This is a single fixture and a
single native resource observation, not a speed comparison with the example
project. Sampled C++ and PyTorch outputs need not match under the same seed.

The connected native PE-to-image run also completed at 512x384 with the same
saved initial latent as the official image reference. It passed 24/25 tensor
checks and the PNG checks (MAE 0.002487, maximum difference 2 on a 0..255 scale).
The decoded tensor's maximum error, 0.01110548, exceeds its fixed 0.01099488
limit, so the complete quality verdict remains failed. See the
[feature delivery report](../artifacts/2026-09-06/features-and-structure/README.md)
for this run, the long-prompt failures, original gates and source snapshots.

Trace adds `input-prompt.txt`, `enhanced-prompt.txt` and `pe-ids.txt` when PE is
enabled. `prompt.txt` always contains the actual text passed to the image
encoder. PE cannot be combined with precomputed `--embeddings`, because those
features would no longer be tied to the original prompt.

The 1024 apple fixture passes the complete saved-input official comparison.
The long English fixture has late tensor failures in both precisions, and the
Chinese landscape fixture has tensor and maximum-pixel failures in both.
FP32 lowers the measured image error substantially, but does not guarantee that
every fixed numerical gate passes. The dated delivery report retains all five
runs and their original gates.

For reproducibility across runtimes, provide the exact same saved initial tensor
with `--latent initial.f32`. An equal integer seed does not imply equal random
tensors across C++ and PyTorch. `--trace-dir NEWDIR` saves text conditioning,
predictions, Euler samples, and decoded floats; it adds transfers and file I/O.
Existing output files and trace directories are rejected. Prompts longer than
the selected text bucket are rejected rather than silently shortened to that
bucket. The official tokenizer's 2048-token maximum still applies.

## Reviewed image-to-image package and CLI

Schema-3 packages remain text-to-image compatible when their `encoder` entry is
`unavailable`. The reviewed 512x384 or 1024x1024 encoder can be added while
assembling its matching static source. For the 512x384 instance:

```sh
python tools/package_dynamic_model.py --schema3 \
  --source models/turbo512x384-s2048-portable \
  --encoder outputs/img2img-encoder-512x384-specialized-v1 \
  --output models/turbo512x384-img2img-shared
build/ernie-image --model models/turbo512x384-img2img-shared --verify-model
```

The builder checks the exact encoder graph, weights, official fixture,
specialization record, source manifests, posterior mode, packing and asymmetric
BN constants before copying encoder bytes into the package object store. The
reviewed shapes are exactly 512x384 and 1024x1024, each bound to its own graph and
official/native evidence. Other encoder shapes are rejected. The 1024 encoder
uses the same learned weights and only the reviewed spatial reshapes; its
production strength-zero reconstruction has passed all six tensor boundaries
and PNG max1. The fixed 1024 strength=0.5 development case also passes
all 21 compared tensor boundaries and PNG max1 against its authenticated
official reference. These fixed cases do not close the formal img2img suite.

```sh
build/ernie-image --model models/turbo512x384-img2img-shared \
  --input source.jpg --prompt 'Preserve the composition, watercolor style.' \
  --width 512 --height 384 --resize fit --strength .5 \
  --seed 42 --steps 8 --output outputs/img2img-new.png
build/ernie-image --model models/turbo512x384-img2img-shared \
  --input source.png --width 512 --height 384 --resize crop \
  --strength 0 --output outputs/reconstruction-new.png
```

Without `--resize`, input dimensions must equal the selected model dimensions.
`stretch` changes both axes, `fit` preserves aspect ratio and adds centered bars
(black unless `--background` is supplied), and `crop` preserves aspect ratio
with a centered crop. Alpha composition defaults to white; `--background
#RRGGBB` changes it explicitly. Strength zero performs only VAE encode/decode
and accepts no prompt, PE, embeddings or text reduction. Positive strength uses
the saved `--latent` as noise when supplied and executes a suffix of the original
full schedule. A trace stores the transformed RGB, preprocessing metadata,
encoder boundaries, saved noise and absolute schedule step filenames. The
15-case image-to-image acceptance matrix remains pending. Current passing
development evidence covers 512x384 and 1024x1024 at strength zero and .5.
The 1024 strength-one run completes all eight steps but passes only 24 of 29
tensor gates: predictions 3, 4, 5, 7 and the decoded tensor fail their fixed
maximum-error gates. Its passing PNG gate does not override those failures.
These fixed inputs do not establish acceptance across the full matrix.

The build workflow checks CPU and software Vulkan kernels and the corruption/relocation
contracts without downloading weights. It has been prepared locally; a GitHub
Actions run has not been performed for this change. Real-weight quality acceptance runs
locally with the pinned official components and saved tensors. See the dated
artifact reports for the precise tested prompts, dimensions, and limitations.


## Generation diagnostics

A separate Linux Vulkan build can add
`-DERNIE_ENABLE_ALLOCATION_METRICS=ON` to the build configuration above.
Use a fresh build directory, for example `build-profile`, and leave
`ERNIE_INSTALL_SDK=OFF`. Configuration requires Python to create an authenticated
copy of the pinned ncnn source with allocator hooks; native inference still
does not need Python. The diagnostic build is separate from the installed SDK.

With a compatible fixed package, request a new JSON report path:

```sh
build-profile/ernie-image --model models/turbo512x384-s2048-portable \
  --width 512 --height 384 --prompt 'A red apple on a wooden table.' \
  --precision fp32 --output outputs/profile-new.png \
  --metrics-json outputs/profile-new.json
```

`total.peak_bytes` records the simultaneous live Vulkan device-memory peak
observed through ncnn allocators. It does not include every driver allocation
or the complete GPU. `stage_times` provides partial host intervals, including a
combined `read_prepare` interval: model reading, weight expansion, pipeline
creation, internal uploads and waits cannot yet be separated there. GPU times,
unobserved transfer byte counts and CPU RSS remain `null`. The JSON states its
coverage and eligibility explicitly; adding `--trace-dir` changes the workload.
Use an uninstrumented, trace-free paired protocol for formal speed comparisons.

The fixed 64x64 instrumentation ON/OFF experiment has completed actual
eight-step generation. All 27 trace files and the PNG are byte-identical, with
all 25 FP32 tensors complete and finite; every observed allocator is released.
See the [execution evidence](../artifacts/2026-09-06/execution-metrics-pipeline64/README.md)
for exact identities, measured scope and limitations. It validates preservation
of that fixture, without adding an official quality or performance result.

## Experimental mapped model loading

`-DERNIE_EXPERIMENT_MAPPED_MODEL_LOADING=ON` requests the pinned ncnn's
file mapping loader for the image pipeline's text encoder, DiT and VAE. The
option is **OFF by default** and does not change precision or model equations.
Each `ncnn::Net` owns its mapping until destruction; ncnn can fall back to its
ordinary file reader when mapping is unavailable. The separate prompt enhancer
does not use this option.

Add the option to a separate build configuration and rebuild `ernie-image`.
The usual model checksum verification still runs before inference. Mapping
does not eliminate BF16-to-FP32 expansion or Vulkan weight uploads.

One fixed 64x64, eight-step FP32 diagnostic observed 64 mapped model files and
retained all 27 trace files and the final PNG byte for byte. Its CLI interval was
275.76 seconds versus 373.09 seconds for the saved preceding non-mapped run,
a 26.1% reduction in this single observation. This was a trace-on internal
comparison with uncontrolled file-cache state, not a formal paired benchmark.
The mapped run reached its 10 GiB cgroup limit and recorded 466,562 `max`
events in `memory.events` without OOM; these page-cache-inclusive measurements do not establish
lower process RSS. Vulkan allocation counts and peak were unchanged. Keep the
option experimental pending representative quality, resource and paired timing
checks. See the [actual mapped-loading records](../artifacts/2026-09-07/mapped-model-loading/README.md).

A subsequent shared-package 512x512 FP32 run completed with all 25 tensors
(4,981,760 elements) and the PNG byte-identical to the previously verified
installed baseline. The supervisor interval was 309.79 seconds versus 466.87
seconds for that baseline; 61 read-only model mappings were observed. The
16 GiB cgroup recorded no `max`, OOM or OOM-kill events. Its sampled peak
`memory.current` was 16,591,130,624 bytes, including file cache; a separate
in-flight RSS observation is not a whole-run RSS maximum. The two runs did not
control file-cache state, so the timing is a diagnostic observation. The build
option remains OFF by default. Exact inputs, binaries, resource records and
comparisons are in the [512x512 evidence](../artifacts/2026-09-07/runtime-rectangles-and-mapped512/README.md).

## Local Linux offline delivery checks

`tools/check_release.py` verifies a locally built runtime archive against its
complete file inventory, installs it under a moved path containing spaces and
Chinese characters, and runs a frozen development case in a separate network
namespace. The inference command uses the native executable; Python supervises
it outside the isolated process. The original source checkout is hidden and the
closed model tree is bound read-only at a new path. This checks relocation by
mounting, without renaming or copying the original model directory.

Prepare with `python3 tools/check_release.py --prepare CASE.json --output NEWDIR`,
where NEWDIR is outside the source repository. The case binds the archive,
inventory, model manifest, prompt/image/noise, expected native PNG and resource
limits by exact identities. Preparation extracts regular enumerated files only,
checks their complete hashes, and restricts tar metadata before interpreting it.
It records a frozen copy of the checking tool. Run that copy with
`python3 NEWDIR/check_release.py --run NEWDIR` inside the exact memory/swap cgroup
specified by the case. Existing run directories cannot be reused for execution.

The current check supports the reviewed 512x384 and 1024x1024 development shapes,
FP32 Vulkan denoising and CPU direct VAE. It verifies the model before and after
generation, records real help/diagnose and missing-model behavior, confirms an
explicit unavailable-driver diagnostic, and requires the output PNG to match
the frozen native PNG byte for byte. Existing output must be refused and remain
unchanged. The supervisor samples the entire cgroup and host memory at 50 ms;
these measurements are neither exact GPU allocation peaks nor a speed test.

With a missing or unusable Vulkan driver, `--diagnose` reports `gpu_count=0`,
`default_gpu_index=-1` and a `vulkan_error` reason while still reading requested
model configuration metadata. An explicit Vulkan generation request continues
to fail. This behavior has actual CPU/Vulkan regression and relocated Linux
installation evidence in `artifacts/2026-09-06/offline-linux-diagnostics/`.

The fixed 1024 strength=0.5 case has now completed actual offline generation
with the installed Vulkan archive. All nine command outcomes match the frozen
case, the PNG is byte-identical to the earlier native fixture, and model
verification passes before and after generation. See the
[execution evidence](../artifacts/2026-09-06/offline-linux-generation/README.md).
The final model verification reached the scope's 10 GiB memory limit without
OOM; this observation is retained and is not a memory-advantage claim. The
checker also has 16 small tests, including a real namespace isolation test.
A passing local case never changes the release draft's redistribution or
publication status; Windows, macOS, public download and the remaining delivery
cases need their own evidence.

The tested archive's executable also has an authenticated
[final link-map inventory](../artifacts/2026-09-06/linked-binary-dependencies/README.md).
Relinking the frozen objects to produce that map yields the exact same binary.
Its selected archive members and final input-section contributions are recorded
separately from dependency availability and redistribution decisions.

## Adaptive DiT weight placement

Vulkan generation defaults to `--dit-weights auto`. Before loading each DiT
input head, block and output head, it queries the compute heap's driver budget
and current process usage. It requests RAM weights when estimated remaining
headroom is smaller than the estimated weight payload plus the configured
reserve. The existing large-sequence preference (more than 6144 total tokens)
also requests RAM. This token count is unrelated to the test supervisor's
6144 MiB whole-device guard.

```sh
# Query the driver before each DiT component; leave 512 MiB extra headroom.
./build/ernie-image --model MODEL --prompt "A red apple" --output auto.png \
  --dit-weights auto --gpu-reserve-mib 512

# Explicit RAM placement; computation still runs on Vulkan.
./build/ernie-image --model MODEL --prompt "A red apple" --output ram.png \
  --dit-weights host
```

`--dit-weights device` requests GPU weights regardless of the automatic policy.
`--gpu-reserve-mib` is extra headroom, not a GPU usage cap. Its default 512 MiB
is a configurable engineering setting, not a model requirement. The payload
estimate is twice the stored weight file size, covering expansion of the
reviewed BF16 files to FP32; it is not a bound on total inference memory.

The query uses `max(heapBudget - heapUsage, 0)` from
[`VK_EXT_memory_budget`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html).
These are driver estimates that can change as other applications run. The
pinned ncnn `get_heap_budget()` reports the budget without subtracting usage,
so it is not used as a free-memory counter. When the extension is unavailable,
auto keeps the existing shape preference and records the missing budget;
explicit host/device placement remains available.

Placement changes happen before a new component loads, after preceding work
has completed. GPU activations and workspace still need device memory. This
does not implement activation spilling, recovery after a failed Vulkan
command, or a retained RAM cache of all 36 blocks. ncnn can itself fall back
from host allocation to device allocation, so the CLI/API report placement
**requests**, not an allocator-level guarantee. Detailed tracing adds
`weight-placement.txt`, with each request's reason, remaining-byte estimate,
weight-byte estimate and reserve. Normal generation reports GPU/RAM request
counts without tensor downloads.

The standard-library-only C++ API exposes `GenerationRequest::dit_weights`
and `gpu_reserve_mib`, plus request counts in `GenerationResult`. Explicit
DiT placement controls require a Vulkan generation device; CPU text and VAE
placement remain controlled by their existing settings.

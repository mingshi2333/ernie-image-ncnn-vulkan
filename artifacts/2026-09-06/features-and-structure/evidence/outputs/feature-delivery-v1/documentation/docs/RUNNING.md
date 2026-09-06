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
when enabled. Applications can link the source-tree target `ernie::pipeline`
and include `ernie/pipeline.h`; generation returns RGB pixels and uses a progress
callback, without requiring command-line parsing or libpng. See
`docs/CODE-STRUCTURE.md` in the source repository.

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

The build workflow checks CPU and software Vulkan kernels and the corruption/relocation
contracts without downloading weights. It has been prepared locally; a GitHub
Actions run has not been performed for this change. Real-weight quality acceptance runs
locally with the pinned official components and saved tensors. See the dated
artifact reports for the precise tested prompts, dimensions, and limitations.

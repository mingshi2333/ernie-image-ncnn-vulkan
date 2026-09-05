# Run the native Turbo generator

The first supported configuration is Linux, ERNIE-Image-Turbo, batch one,
8 Euler steps, CFG 1, and prompt expansion disabled. Model dimensions and
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

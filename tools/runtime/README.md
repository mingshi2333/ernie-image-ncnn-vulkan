# ERNIE-Image-Turbo

Extract this entire folder, then open a terminal here. You need Python 3.10+
for the download helper and an up-to-date graphics driver for Vulkan.
The C++ program performs inference without Python packages, PyTorch or CUDA.

```sh
python3 run.py --prompt "A red apple on a wooden table, soft daylight." --output apple.png
```

On Windows, use `python` instead of `python3`. The first generation downloads
and verifies the 23.27 GB main model into `models/turbo/`. Interrupted downloads
resume on the next invocation. The default size is 512 x 512, with 8 steps and
Vulkan FP16. Existing output files are preserved; choose a new filename to run again.

```sh
python3 run.py --prompt "A mountain lake at sunrise." --width 1376 --height 768 --precision fp32 --output lake.png
python3 run.py --prompt-file prompt.txt --width 768 --height 1024 --precision fp16 --output portrait.png
```

Both dimensions must be multiples of 16, from 16 to 2048, with at most
2,097,152 pixels in total. The same model supports all these sizes and
`--precision fp32|fp16|bf16`. BF16 remains experimental. Precision selects the
DiT execution path; it does not make every stage of the pipeline 16-bit.

Useful commands:

```sh
python3 run.py --diagnose
python3 run.py --download-only
python3 run.py --verify-model
python3 run.py --with-pe --prompt "A small village." --output village.png
python3 run.py --model /path/to/existing/turbo --prompt "A red apple." --output reuse.png
python3 run.py --models-dir /path/to/model-storage --prompt "A red apple." --output moved.png
python3 run.py --help-all
```

`--with-pe` downloads an additional 7.68 GB optional prompt enhancer. It runs on
the CPU and is unnecessary for ordinary text-to-image generation. `turbo/objects/`
and `pe/block-*/` are the model's internal storage; retain their names and structure.
The runtime loads all required blocks automatically.

Once the model is downloaded, generation works offline. You can also invoke
`bin/ernie-image` directly with `--model`, `--width` and `--height`. On Linux,
set `LD_LIBRARY_PATH` to this folder's `lib/`; on macOS, set `VK_DRIVER_FILES`
to `vulkan/MoltenVK_icd.json`. The Python launcher sets these paths automatically.

The Linux build targets Ubuntu 24.04 x86_64 (glibc 2.39, GCC 13 C++ runtime).
Windows targets Windows 10/11 x64. The macOS build targets macOS 15 and its
architecture is recorded in the archive name; it is ad-hoc signed, not notarized.
Vulkan drivers are supplied by the operating system on Linux/Windows; MoltenVK
is bundled on macOS. CPU execution is available with `--device cpu`.

`build-info/` contains the source revision, ncnn revision and native framework
CI results. `files.json` contains checksums of the unpacked files, and the adjacent
`.sha256` download authenticates the ZIP. Runtime and third-party licenses are
included under `licenses/`; separate model terms are under `manifests/`.

[Source, images and numerical results](https://github.com/mingshi2333/ernie-image-ncnn-vulkan)

[Model downloads](https://huggingface.co/akashimio/ERNIE-Image-Turbo-ncnn)

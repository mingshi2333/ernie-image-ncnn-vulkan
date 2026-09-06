# F2 img2img start and denoiser schedule contract

This slice implements the already specified image-conditioned latent initialization; it does not enable image generation from an input image before encoder/package integration and actual quality validation.

`make_img2img_start` accepts the patchified, normalized encoder mean and saved FP32 CHW noise. Strength zero returns independent encoded storage with start_step=steps and consumes no noise. Positive strength uses the pinned peer's positive round-half-up rule, with at least one denoising step, and combines sigma*noise + (1-sigma)*encoded. Strength one returns an independent byte-preserving noise clone. The implementation checks layout, finite values, matching shapes, finite strength, thread bounds and descending schedule endpoints. Arithmetic uses the existing scheduler's no-FMA-contraction compilation policy.

Both CPU and Vulkan `denoise` overloads now accept an optional trailing start_step, defaulting to zero for existing callers. They keep the original full Turbo schedule and use absolute step indices for callbacks, with statistics only for the executed suffix. A start equal to steps performs no DiT model execution; the CPU contract proves this using nonexistent model paths and verifies no observer call or stale statistics.

Validation: current `ernie-image` builds, and five focused CTests pass (img2img start, original latent arithmetic, public API, request validation, CLI contract). New cases cover 0/.25/.5/.75/1, an exact half step where bankers rounding differs, the smallest positive FP32 strength, independent output storage, malformed/nonfinite requests, skipped denoising, and invalid start indices.

Not yet claimed: actual VAE encoder parity, full 15-case img2img, active pipeline/CLI input support, Vulkan suffix numerical equality, or native package encoder completeness. The tiny schedule tests cannot close those requirements. Encoder BN epsilon=1e-4 and decoder inverse-BN epsilon=1e-5 remain separate contracts pending the independent official-module experiment.

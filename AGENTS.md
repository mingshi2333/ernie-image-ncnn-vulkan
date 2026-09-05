# Project instructions

## Scope and status

- This is the standalone ERNIE-Image-Turbo ncnn/Vulkan project. Do not modify sibling ncnn or pnnx workspaces as a side effect.
- Start with `README.md`, `sources.lock.json`, `docs/ROADMAP.md`, and the latest artifact report.
- The experimental native generator has passed a complete 64x64 prompt-to-PNG comparison against staged official modules. A native 1024x1024 run also completed on this host; it is a functional/resource smoke without full-resolution official denoising parity. Read the latest pipeline artifact for limits; small images are numerical fixtures, not prompt-quality evidence.
- Use Chinese for progress and technical reports unless the user asks otherwise.

## Evidence and correctness

- Pin upstream revisions. A feature in a local modified ncnn checkout is not evidence that upstream has it.
- Preserve official ERNIE mathematics: tokenizer behavior, hidden-state selection, YaRN, DiT RoPE, shared AdaLN, GELU gating, latent packing, BN normalization, and sigma schedule.
- Keep CPU and Vulkan cache handles opaque. CPU extraction uses `type=1`. Use a distinct session cache allocator, consume-and-replace handles, and release caches before allocator destruction. Do not use shallow copies as independent session snapshots.
- Do not reuse DiT K/V across denoising steps as an exact optimization. Approximate caching needs a separate opt-in experiment and quality gate.
- Use identical saved latents and embeddings for full-model parity. An identical integer seed across frameworks is insufficient.
- State whether a result is source inspection, synthetic operator validation, real-weight validation, or an end-to-end benchmark.
- Do not claim speedup from buffer counts or from changing model variants, step counts, precision, or PE settings.
- Use `tools/prepare_block.py` to retain the reviewed ERNIE GELU/RoPE semantics. The pinned upstream Vulkan GELU uses a tanh approximation; standard RotaryEmbed assumes a shared half-width angle table.
- Register the project's RMSNorm and final LayerNorm wrappers. Native FP16 intermediate squares can overflow; the wrappers promote normalization on device. Final LayerNorm is non-affine, width 4096, and conditioning has no SiLU.
- Real text causes DiT residual activations to exceed 65504. FP16 inference requires both residual sums to use ErnieResidualAdd and retain the FP32 skip path. Normalized projections return to model storage precision. Never silently clamp the residual.
- The FP32 Euler master latent is checked after each step. Vulkan downloads only 128 status floats for this check; tracing downloads activations separately and changes measurement scope.
- The pinned Mistral3 text configuration dispatches MistralModel, not Ministral3. Required hidden_states[-2] is block 24 output: 25 blocks, without final norm. The initial text bucket is 32 tokens.
- CPU VAE uses FP64 mean/centered-variance reductions with FP32 activations and affine arithmetic. Large native GroupNorm reductions failed the fixed numerical gate before this correction.
- BF16 standalone gates fail for blocks 31, 33, and 35 on the current synthetic fixture. Do not erase those failures or relax gates after observing results. Connected predictions are a different fixture scope.
- Treat BF16 ModelBin packing as lossless on-disk storage only when the full reconstructed FP32 stream hash matches. Loading still expands data and does not establish a peak memory reduction.
- Export a separate static graph for each token bucket. A matching operator list does not prove shape compatibility.
- VAE spatial specialization is limited to the full-hash-reviewed flatten/unflatten pair, and requires an independently executed target-resolution official fixture. Preserve the failed high-resolution whole-graph pnnx conversion; do not re-run its unbounded memory expansion.
- Record failures and skipped tests. Avoid repeating passing checks without a change or unresolved concern.

## Resource use and delivery

- The initial target is an 8GB GPU and 32GB system RAM. Inspect current availability before large runs. Begin with one block and staged weight loading.
- Keep weights, build products and generated images out of Git. Store small manifests, measurements and source hashes in `artifacts/`.
- Snapshot runner binaries before long validation jobs. A concurrent build can temporarily remove or replace its executable; the first batch recorded this failure.
- Run large GPU jobs sequentially. Overlapping a VAE test and DiT generation caused allocation failures and a SIGSEGV; that run does not establish an isolated-device capacity limit.
- External publication, push and release require user authorization. Local construction and tests are within the approved project scope.

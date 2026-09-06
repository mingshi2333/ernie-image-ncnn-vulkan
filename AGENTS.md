# Project instructions

## Scope and status

- This is the standalone ERNIE-Image-Turbo ncnn/Vulkan project. Do not modify sibling ncnn or pnnx workspaces as a side effect.
- Start with `README.md`, `sources.lock.json`, `docs/ROADMAP.md`, and the latest artifact report.
- Current implementation plan: `docs/superpowers/plans/2026-09-06-surpass-reference.md`, with six stage plans and `surpass-acceptance.json`. Read `.superpowers/sdd/2026-09-06-surpass-reference/progress.md` for current ownership, measured evidence and remaining work. Targets are not results; preserve historical failures and do not declare superiority until every required gate has evidence.
- Latest complete development evidence: `artifacts/2026-09-06/shared-native-pipeline/README.md`. The reviewed 512x384 shared instance ran actual CPU greedy PE, all 25 native text layers with explicit Vector reduction, eight Vulkan FP32 steps and CPU VAE: 25/25 tensor gates, exact PE text/IDs, PNG max1, and all tensors/PNG bitwise equal to the preceding fixed-package native run. This single fixture does not close broad quality, dynamic shapes, performance or platform gates. The original compensated PE/apple 24/25 decoded failure remains historical evidence in `artifacts/2026-09-06/features-and-structure/README.md`.
- Fixed 1024x1024 encoder and production strength-zero reconstruction have independent actual evidence: three encoder and six reconstruction boundaries pass, PNG max1. See `.superpowers/sdd/2026-09-06-surpass-reference/task-F2-strength0-1024-review.md`; fixed strength.5 now also passes all21 compared tensor boundaries and PNGmax1, independently recomputed in task-F2-positive-1024-execution-review.md. Formal15 remains unrun. Linux CPU/Vulkan installed C++ consumption is recorded in `artifacts/2026-09-06/installed-cpp-sdk/README.md`; this is API/installation evidence, not full-model or Windows/macOS acceptance.
- Fixed 1024 strength1 completes all8 steps, but remains quality_gate_failed:24/29 tensor gates pass, predictions3/4/5/7 and decoded fail fixed maximum-error gates; PNGmax2 passes. Full independent review f21ad9f preserves the negative result. Do not infer broad img2img quality from the passing strength0/.5 fixtures. Linux missing-driver diagnostics are fixed and actual CPU5/5/Vulkan6/6 affected CTests pass; new full-model offline execution is tracked separately in the ledger.
- Installed Linux archive now passes the actual fixed 1024 strength.5 offline case with the complete source tree hidden, network disabled and a Chinese-space model path. All nine command outcomes and the exact native PNG match; see `artifacts/2026-09-06/offline-linux-generation/README.md`. Final model verification reaches the 10GiB scope limit with max49634 but no OOM, so do not claim whole-chain max0, a 6.49GB whole-chain peak or S/M acceptance. Other delivery cases and platforms remain open.
- The experimental native generator has passed complete 64x64 and 1024x1024 apple comparisons against staged official modules. After FP32 attention compensation and query chunking, the 40-token English fixture passes 24/25 tensors and PNG (MAE 0.00268, max 1); prediction-7 still fails its maximum-error gate. Chinese compensation passes 21/25 tensors; PNG MAE 0.02263, max 13 still fails. Read artifacts/2026-09-06/attention-parity/README.md and docs/NUMERICAL-DIAGNOSTICS.md for current limits. Historical low-precision failures remain; do not call this broad quality acceptance.
- Use Chinese for progress and technical reports unless the user asks otherwise.

## Code organization

- Read docs/CODE-STRUCTURE.md. Public generation types live in include/ernie/pipeline.h; cli/ owns arguments and PNG I/O; src/pipeline.cpp owns stage orchestration. Keep model math and cache sessions in their components.
- Keep the public header independent of ncnn/PNG headers. The pipeline returns RGB pixels and reports progress through callbacks.
- Root CMake delegates dependencies, tokenizer, runtime, CLI, probes and tests to their own directories. Keep documented executable paths in the build root.
- tools/source_inventory.py must cover all source and build directories in new evidence. Preserve old snapshots and command compatibility.

## Evidence and correctness

- Pin upstream revisions. A feature in a local modified ncnn checkout is not evidence that upstream has it.
- Preserve official ERNIE mathematics: tokenizer behavior, hidden-state selection, YaRN, DiT RoPE, shared AdaLN, GELU gating, latent packing, BN normalization, and sigma schedule.
- Keep CPU and Vulkan cache handles opaque. CPU extraction uses `type=1`. Use a distinct session cache allocator, consume-and-replace handles, and release caches before allocator destruction. Do not use shallow copies as independent session snapshots.
- Optional PE is actual Ministral3: 26 layers, final norm and tied LM head, with a distinct PE tokenizer/chat template. Greedy CPU FP32 315-token EOS and every logit passed the fixed 2e-4 gates. Do not confuse this with the 25-layer image text encoder.
- PE input/output limits are 2048 each; native cache capacity is at most 4096, below the official 16384-position query-scaling boundary. Prefill is sequential, without padded cache entries. Same sampled seed across C++ and PyTorch is not a parity oracle.
- The 512x384, 1080-token image fixture runs to completion in FP32 and BF16 but fails the fixed full-quality gates (19/25 and 11/25 tensors; PNG max 17 and 255). Keep BF16 experimental.
- Do not reuse DiT K/V across denoising steps as an exact optimization. Approximate caching needs a separate opt-in experiment and quality gate.
- Use identical saved latents and embeddings for full-model parity. An identical integer seed across frameworks is insufficient.
- State whether a result is source inspection, synthetic operator validation, real-weight validation, or an end-to-end benchmark.
- Do not claim speedup from buffer counts or from changing model variants, step counts, precision, or PE settings.
- Use `tools/prepare_block.py` to retain the reviewed ERNIE GELU/RoPE semantics. The pinned upstream Vulkan GELU uses a tanh approximation; standard RotaryEmbed assumes a shared half-width angle table.
- Register the project's RMSNorm and final LayerNorm wrappers. Native FP16 intermediate squares can overflow; the wrappers promote normalization on device. Final LayerNorm is non-affine, width 4096, and conditioning has no SiLU.
- Real text causes DiT residual activations to exceed 65504. FP16 inference requires both residual sums to use ErnieResidualAdd and retain the FP32 skip path. Normalized projections return to model storage precision. Never silently clamp the residual.
- The FP32 Euler master latent is checked after each step. Vulkan downloads only 128 status floats for this check; tracing downloads activations separately and changes measurement scope.
- The pinned Mistral3 text configuration dispatches MistralModel, not Ministral3. Required hidden_states[-2] is block 24 output: 25 blocks, without final norm. Independently exported text buckets are 32, 64 and 2048 tokens, including BOS. The 1080-token official-weight text fixture passes at NRMSE 7.70e-6.
- CPU VAE uses FP64 mean/centered-variance reductions with FP32 activations and affine arithmetic. Large native GroupNorm reductions failed the fixed numerical gate before this correction.
- CPU VAE defaults to direct convolution with Winograd disabled. The 1024 standalone fixture passes at NRMSE 9.41e-7 and 5.42 GiB peak RSS; the complete apple run uses 5.82 GiB. The old SGEMM mode remains explicit for comparisons.
- Single-step teacher forcing is a diagnostic, not a substitute for free-running parity. Step 6 of the long FP32 fixture passes with official input tensors; the full trajectory still fails. Do not loosen existing gates or discard either result.
- BF16 standalone gates fail for blocks 31, 33, and 35 on the current synthetic fixture. Do not erase those failures or relax gates after observing results. Connected predictions are a different fixture scope.
- Treat BF16 ModelBin packing as lossless on-disk storage only when the full reconstructed FP32 stream hash matches. Loading still expands data and does not establish a peak memory reduction.
- Export a separate static graph for each token bucket. A matching operator list does not prove shape compatibility.
- VAE spatial specialization is limited to the full-hash-reviewed flatten/unflatten pair, and requires an independently executed target-resolution official fixture. Preserve the failed high-resolution whole-graph pnnx conversion; do not re-run its unbounded memory expansion.
- FP32 non-Flash attention uses compensated softmax/P@V reductions and at most 128 query rows per chunk, retaining all K/V. Check both pinned shader SHA256 values; never bypass a hash failure when updating ncnn. Chunk completions count as compute submissions and do not download activations. Preserve native cache and low-storage Flash paths.
- Reference-embedding runs bypass the native text encoder and are diagnostics even if they pass. The CPU FP64 RMSNorm candidate was rejected: its first-layer gain did not improve the complete text encoder. Do not promote local operator improvements to full-model acceptance.
- Record failures and skipped tests. Avoid repeating passing checks without a change or unresolved concern.

## Resource use and delivery

- The initial target is an 8GB GPU and 32GB system RAM. Inspect current availability before large runs. Begin with one block and staged weight loading.
- Keep weights, build products and generated images out of Git. Store small manifests, measurements and source hashes in `artifacts/`.
- Schema-2 portable packages contain all 136 runtime files with no internal symlinks. Both native and Python checks cover every runtime checksum, file size, revision and model.cfg agreement. Keep schema-1 development package compatibility.
- Snapshot runner binaries before long validation jobs. A concurrent build can temporarily remove or replace its executable; the first batch recorded this failure.
- Run large GPU jobs sequentially. Overlapping a VAE test and DiT generation caused allocation failures and a SIGSEGV; that run does not establish an isolated-device capacity limit.
- The isolated full-matrix FP32 compensation long run also hit allocation failure and SIGSEGV during step 3. Preserve it separately from the earlier overlapping-job failure. Query chunking subsequently completed all eight steps with a 4098 MiB whole-device sampled peak; this is not an allocator peak guarantee.
- External publication, push and release require user authorization. Local construction and tests are within the approved project scope.

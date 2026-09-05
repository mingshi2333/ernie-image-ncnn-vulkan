# Project instructions

## Scope and status

- This is the standalone ERNIE-Image-Turbo ncnn/Vulkan project. Do not modify sibling ncnn or pnnx workspaces as a side effect.
- Start with `README.md`, `sources.lock.json`, `docs/ROADMAP.md`, and the latest artifact report.
- Current code validates a real DiT block on CPU/Vulkan, including 4160-token inputs. It is not a working image generator. Preserve this distinction in status reports.
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
- Treat BF16 ModelBin packing as lossless on-disk storage only when the full reconstructed FP32 stream hash matches. Loading still expands data and does not establish a peak memory reduction.
- Export a separate static graph for each token bucket. A matching operator list does not prove shape compatibility.
- Record failures and skipped tests. Avoid repeating passing checks without a change or unresolved concern.

## Resource use and delivery

- The initial target is an 8GB GPU and 32GB system RAM. Inspect current availability before large runs. Begin with one block and staged weight loading.
- Keep weights, build products and generated images out of Git. Store small manifests, measurements and source hashes in `artifacts/`.
- External publication, push and release require user authorization. Local construction and tests are within the approved project scope.

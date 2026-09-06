# SDD ledger — implementation in progress

Plan: docs/superpowers/plans/2026-09-06-surpass-reference.md
Worktree: /home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference
Branch: codex/surpass-reference. Base b9eb755. User authorized parallel implementation.
F/Q/S/M/D/A gates remain open. No superiority claim or external push/release.

## Completed implementation and measured development evidence

- B1: 101 frozen cases (72 formal, 8 development, 6 performance, 15 img2img); immutable input hashes and independent review fixes in c8835ee. Formal72 and formal15 remain unrun/unrevealed.
- B3: e70fb7a binds six cases, five paired fresh-process repetitions, AB/BA, precision, model/input hashes and failure semantics; 3b3f971 makes unit tests independent of ignored local output. No actual S/M comparison yet.
- B2: peer built with its own pinned ncnn. All28 fixed-revision asset files now present and verified in outputs/reference-port-v1/assets-manifest.json; downloader no longer active. Weight audit 13b6a75 covers all254 VAE records, 251 direct matches plus3 bitwise-derived constants, and16/16 named MHA projections. Source-only capacity review finds optimistic21.23GiB persistent BF16 weights/42.45GiB FP32; this is not measured RSS/OOM and cannot establish a speed win. No original peer full image yet.
- Q1: 8e4430f/c64ff31 preserves crossed decoder and trajectory findings. Decoder-only <=7.153e-6 passes; historical full native PE/apple 24/25 decoded .01110548 > .01099488 remains recorded.
- Q2: explicit Vector down-reduction remains opt-in. 0b677d7 seals a REAL complete native greedy PE -> all25 text blocks -> eight Vulkan FP32 steps -> CPU direct VAE at512x384: 25/25, exact315-token PE text/IDs, PNG max1/MAE.0015598. All25 saved tensors match the earlier saved-vector diagnostic bitwise. Native runner ae04f103... predates F1 bridge. No general numerical closure: Chinese Vector22/25+PNGmax13, some downstream errors worse than compensated21/25 baseline. English/1080-token/low-storage historical failures remain.
- Q2 Chinese: fefb93f identifies step6 trajectory amplification, not a proven operator cause. 9c17c33 same-official-input step6 teacher forcing passes NRMSE1.23585e-5/max.000442982. 363ce43 swaps only native time features: max.000365376 still passes, so this difference does not reproduce the large failure. 0aeea71 exactly replays the actual native step6 output using all six original native inputs; SHA622738cf...f3fe7, zero differing floats. Full-run22/25 and PNGmax13 remain failed. Latent-only/text-only conditional sensitivity is the next diagnostic, not a new quality gate.
- Q3: f73cb5a seals all12 PE development cases, exact text/IDs and independently recomputed1742 logits arrays (each131072 values). Longest actual output526 EOS; near-capacity input2048+output32. Not a2048-output/4096-cache stress proof. Old315-token parity retained.
- F1: reviewed ShapePlan/schema3 CAS connects ModelPackage/ComponentFiles into all native text/DiT/VAE paths without temporary param/weight copies, preserving legacy probes and math. Real shared512 full native PE/text/DiT/VAE passes25/25 and PNG; all25 tensors plus PNG are bitwise equal to the preceding fixed-package run. Artifact artifacts/2026-09-06/shared-native-pipeline/README.md records the original execution and corrected independent audit separately. a77dccc authenticates the entire reviewed oracle and complete25-boundary denominator; independent review ded0f14 closes all Important findings. Only known512x384/s2048 and1024x1024/s64 instances; shared1024 actual generation and arbitrary dynamic shape validation remain open.
- F2: direct CPU FP32 encoder and independently validated512 reconstruction retain encodeBN1e-4/decodeBN1e-5. 638a58c/6a30dae connect the trusted optional512 encoder to schema3/API/CLI; f04b35f fixes source-membership and bilinear edge errors, independently closed in c7897da. b07d9ed records the ACTUAL production strength0 path: encoder/decoder only after full80-object verification, no PE/text/DiT/noise, six boundaries pass, decoded max1.65403e-5, PNGmax1/MAE.000125461. 4edcfb3 adds the separately authenticated official positive suffix oracle. Positive strength0.5 native comparison below is running; only reviewed512x384 encoder is available,1024 remains unavailable.
- F3: f5d1778 adds bounded PNG/JPEG/BMP/TGA I/O, gray/RGBA/background and Unicode paths; real roundtrip/error tests pass. Public header remains stdlib-only; image input integration joins F2.
- O1: opt-in actual VkDeviceMemory hook/session with authenticated candidate/peer derived ncnn copies; defaultOFF preserves original source/dependencies. Independent ON/OFF builds complete. 847664c records actual GPU proof after preserving the v1 malformed float-param failure: v2 outputs1024 bytes bitwise equal; ON observes11 allocations, peak2134016 bytes, final live0 afterdestroy, allallocators inactive, including actual importedhost memory. OFF is unavailable/null. This is a small probe; full-model profile integration and paired process allocation peaks remain pending.

## Current ownership and execution records

- root: B2 full authenticated weight scan, F1 evidence persistence and source documentation; global GPU/resource serialization and ledger.
- paired_metrics: F2 positive strength0.5 same-input native/official development comparison. Owns pipeline img2img branch and dedicated oracle. GPU slot currently allocated.
- frozen_corpus: Q2 exact replay complete; preparing two conditional latent-only/text-only sensitivity experiments with native constants/time fixed. CPU only until scheduled.
- reference_adapter: inspecting production CLI/Pipeline/GpuContext lifecycle for truthful opt-in O1 profile integration. No GPU; file ownership assigned before edits.

## Live workloads and safeguards

- B2: session81890 / ernie-b2-fullweight-audit-v6.scope, frozen worker outputs/reference-port-v1/weight-audit-all-v6-execution. CPU-only3GiB/swap0/200% scope scans all28 completed peer assets against102 authenticated official components; graph mapping and S are not automatically closed by content matching. First scope-launch attempt failed before process start because user-bus environment was absent; corrected to the existing owned /run/user/1000 bus, leaving frozen command/source unchanged.
- F2: outputs/f2-positive05-512x384-v1. Official steps4/5/6/7 completed; matching native4-step suffix running with original short apple prompt, PEoff, VectorFP32 and savednoise seed20260906. This is development evidence, not formal15 or a performance round.
- Completed and released: F1 native96587/audit76915; Q2 time swap and exact native replay; O1 ON/OFF builds and both realv2 GPU probes. Their frozen execution directories remain authoritative.
- GPU queue: F2 positive suffix -> Q2 conditional sensitivity after CPU preparation/review. Large GPU jobs never overlap.
- build-dev executables are current incremental products, not immutable evidence; freeze binaries/sources before each real experiment. Existing outputs/build/ snapshots retain their own identities.

## Required next work

Complete positive suffix native comparison and independent review; isolate Chinese trajectory sensitivity and integrate truthful full-model metrics. Extend native shapes only with audited graph contracts and independent target-shape fixtures. Close remaining historical full-trajectory failures before default promotion/formal acceptance. Continue reference baseline/capacity work without manufacturing an OOM or speedup from a source lower bound. Complete actual paired speed/memory, platforms/offline delivery and formal quality only after prerequisites pass.

Two independent human blind ratings and actual Windows full-model/offline evidence remain mandatory external evidence. No fabricated review/platform success; they do not prevent useful local work.

All Git mutations use the shared flock .superpowers/sdd/2026-09-06-surpass-reference/git-commit.lock and exact file ownership. Preserve original models, upstream checkouts, old results and all negative evidence.

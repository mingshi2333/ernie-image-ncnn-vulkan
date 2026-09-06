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
- Q2 Chinese: fefb93f identifies step6 trajectory amplification, not a proven operator cause. 9c17c33 same-official-input step6 teacher forcing passes NRMSE1.23585e-5/max.000442982; this does not change full-run22/25 failure. f68b5d6 finds native/official timestep bits identical but228/4096 feature values differ (max1.46627e-5); single-factor GPU swap prepared, not yet run.
- Q3: f73cb5a seals all12 PE development cases, exact text/IDs and independently recomputed1742 logits arrays (each131072 values). Longest actual output526 EOS; near-capacity input2048+output32. Not a2048-output/4096-cache stress proof. Old315-token parity retained.
- F1: 77049c7 reviewed ShapePlan/canonical graphs; 47151b4 implements strict schema3 CAS native/Python protocol. f253d5e connects ModelPackage/ComponentFiles into all native text/DiT/VAE paths without temporary param/weight copies, preserving legacy probes and math. 8ce8c0d/5d7633f independent review passes. Eight freshly relinked CPU/API/CLI/package contracts pass; real native shared package --verify-model passes. Actual full shared image run below is pending. Only known512x384/s2048 and1024x1024/s64 instances; arbitrary dynamic shape validation remains open.
- F2: 1edea8a/bb87a4f implement direct CPU FP32 native encoder and32 reconstruction. accad82/4655b05 execute official512 encoder and strength-zero native reconstruction: six boundaries pass, decoded max1.65403e-5, PNGmax1/MAE.000123766. Only reviewed32x32/64x32/exact512x384 sizes; reversed384x512 rejected. 638a58c adds trusted optional512 encoder CAS inventory while preserving unavailable old schema3 packages. Production CLI/pipeline integration ongoing.
- F3: f5d1778 adds bounded PNG/JPEG/BMP/TGA I/O, gray/RGBA/background and Unicode paths; real roundtrip/error tests pass. Public header remains stdlib-only; image input integration joins F2.
- O1: 48a72a5 foundation and27714e0 opt-in actual VkDeviceMemory hook/session, authenticated candidate/peer derived ncnn copies. DefaultOFF leaves original source/dependencies unchanged. 7783e76 adds real allocator/GPU arithmetic contract; source/patch/fake-backend tests and syntax checks pass. Actual allocation and ON/OFF bitwise GPU proof pending below.

## Current ownership and execution records

- root: F1 shared native end-to-end equivalence and reference contract/validator/auditor; global GPU/resource serialization, evidence and ledger.
- paired_metrics: F2 trusted encoder package + production img2img API/CLI integration and CPU contracts. Owns pipeline img2img branch. Full models/GPU require queue allocation.
- frozen_corpus: independent shared-validator review, then Q2 one-variable timestep feature swap. Two Important validator findings (source binding and incomplete reference denominator) were reproduced in ce5826f; root fix with trusted oracle registry/complete boundary contract is under independent review. The running old snapshot is preserved and will require corrected post-run audit.
- reference_adapter: O1 real ON/OFF ncnn/probe builds in build-o1/, separate3GiB swap0 scopes. ON complete; OFF building. No GPU before root releases slot.

## Live workloads and safeguards

- F1: unified exec session96587; outputs/f1-shared-native-worker-v1/state.json and samples.jsonl are authoritative process/resource records. Output outputs/pipeline512x384-shared-pe-native-vector-fp32-v1. Native image runner6a3481e... with213 frozen source files; separate validation-source215 file snapshot. Native program directly loads shared CAS, real greedy PE and native Vector text. Same saved official reference/noise as0b677d7. Recursive descendants observed from launch,18GiB RSSsum/6GiB wholeGPU/1GiB host reserve and2400s guards; not an allocation peak or performance measurement.
- O1: ON original build unit was frozen during PE loading and resumed afterwards; ON completed. OFF session53153, unit ernie-o1-build-off. Both2 jobs,3GiB memory and swap0; no main build-dev mutations. Read agent report before resuming handles.
- GPU queue: F1 full shared pipeline -> Q2 native-time-only step6 -> O1 real allocation contract. Large GPU jobs never overlap.
- build-dev executables are current incremental products, not immutable evidence; freeze binaries/sources before each real experiment. Existing outputs/build/ snapshots retain their own identities.

## Required next work

Finish F1 actual generation and corrected independent audit; execute queued Q2/O1 experiments; independently review and validate production img2img strength0 and positive suffix against official same-input reference. Extend native shapes only with audited graph contracts and independent target-shape fixtures. Close remaining historical full-trajectory failures before default promotion/formal acceptance. Continue reference baseline/capacity work without manufacturing an OOM or speedup from a source lower bound. Complete actual paired speed/memory, platforms/offline delivery and formal quality only after prerequisites pass.

Two independent human blind ratings and actual Windows full-model/offline evidence remain mandatory external evidence. No fabricated review/platform success; they do not prevent useful local work.

All Git mutations use the shared flock .superpowers/sdd/2026-09-06-surpass-reference/git-commit.lock and exact file ownership. Preserve original models, upstream checkouts, old results and all negative evidence.

# SDD ledger — implementation in progress

Plan: docs/superpowers/plans/2026-09-06-surpass-reference.md
Worktree: /home/mingshi/Project/AI/ernie-image-ncnn-vulkan/.worktrees/surpass-reference
Branch: codex/surpass-reference. Base b9eb755. User authorized parallel implementation.
No superiority claim: F/Q/S/M/D/A gates remain open. No external push/release.

## Completed implementation slices and independent reviews

- B1 frozen corpus: 101 cases (72 formal, 8 development, 6 performance, 15 img2img) and fixed noise/input contracts. Formal 72 remain unrun. c8835ee closes both independent review findings; frozen v1 input hashes unchanged.
- B3 metrics: e70fb7a binds six cases, 5 paired repetitions, AB/BA, precision, model/input hashes, and failure semantics to frozen protocol. Independent round-2 findings closed. 3b3f971 removes accidental ignored-output dependency from unit tests. Measurement implementation is not actual baseline data.
- Q1 crossed VAE: 8e4430f executes 56 hash-verified source snapshots. Four actual CPU decodes in outputs/vae-cross-pe-v2 reproduce history bitwise. Decoder-only <=7.153e-6 passes; upstream latent connected max .01110548 still fails .01099488. Independent review c64ff31 closes Important findings. Full Q1 quality closure remains open.
- Q2 trajectory: 6c50d7f records 25 ordered tensor boundaries and identities. Actual saved official-text diagnostic 25/25 passes while native-text history remains 24/25; bypass run explicitly ineligible for native acceptance. Some NRMSE worsens, so text is not claimed sole cause. Independent review c64ff31 passes.
- Q3 sampling/batch contract: 7931c58 bounded sampler tests and 12 frozen PE development cases. Actual batch below; full acceptance pending.
- F3 image I/O component: f5d1778 PNG/JPEG/BMP/TGA, gray/RGBA/background, bounded input and Unicode paths, real roundtrip/error tests pass. CLI integration in progress; no full F3 claim.
- F1 first slice: 77049c7 checked ShapePlan and full-hash audit of 192 historical static graphs. Dynamic generation, independent portrait/extreme fixtures, schema-3 runtime integration remain pending. Existing 6144 total-token protection preserved.
- build-dev: real Release build with pinned source ncnn, system glslang, tokenizer and native generator all built. Original build/ binaries are historical; do not mistake them for current implementation.

## Active ownership

- root: resource supervision, PE actual batch, B2 compare/adapter integration and later model validation.
- reference_adapter: Q2 text GEMM numerical diagnostics only; no default runtime changes. Batch InnerProduct candidate reproduced Gemm bitwise and is rejected. Testing true K-reduction change with one real down projection.
- paired_metrics: F3 public API/CLI controls and image I/O integration; src/pipeline.cpp only validation/configuration, not math.
- frozen_corpus: F1 strict candidate graph/package contract and independent native template instantiation. No claim that pending schema-3 is a runnable package.

## Live workloads and resources

- Peer downloader PID 1524331, start ticks 8899889, unified exec session 97237. Handoff outputs/reference-port-v1/download-handoff.json. text_encoder/language_model_encoder.ncnn.bin 11639808700 bytes verified; DiT large payload in progress. Do not restart while alive. Full peer first-image baseline still pending assets and resource preflight.
- PE batch driver outputs/pe-development-v1/run_batch.py launched 2026-09-06. Journal records PID/start ticks/commands/source hashes, serialized official and native runs for 12 cases. One full CPU model at a time, 18 GiB process-group RSS and host-availability guard, 1800 seconds per stage. Treat guard stops as incomplete, not OOM or quality failure. Runner and all 58 local Python tools snapshotted; model models/pe-cpu-v1. Check journal and driver.log for current handle/results.
- GPU: no full-model job currently active. Keep large GPU work serialized.

## Next required work

Finish actual PE cases and diagnostics, strict B2 weight correspondence and first complete peer run, numerical closure across historical failed native fixtures, performance/memory optimizations only against matched measured baseline, dynamic shapes/img2img/native delivery, then frozen formal acceptance. Two independent human blind ratings and actual Windows full-model/offline evidence remain mandatory and cannot be fabricated.

All Git mutations use shared flock .superpowers/sdd/2026-09-06-surpass-reference/git-commit.lock and exact file ownership. Preserve original models, old results and all negative evidence.

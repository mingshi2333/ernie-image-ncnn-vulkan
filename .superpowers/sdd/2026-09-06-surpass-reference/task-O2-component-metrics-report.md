# O2 component interval preparation report

## Status

Implemented an opt-in diagnostic child-interval stream underneath the existing allocation/execution metrics collector. No model or GPU run was performed. The records are explicitly nested diagnostics and are excluded from `stage_times` phase sums.

Each record carries `component`, `absolute_step`, `block`, `boundary`, `status`, and host duration. The current boundaries are `net_setup_param`, `model_load_composite`, `extract_compute_composite`, and `net_destroy`. Text blocks, DiT input/output heads, streamed DiT blocks, and VAE decode use the same vocabulary. Absolute denoise schedule indices are preserved for suffix runs.

Collection is disabled unless `generate_with_metrics` supplies a collector. Existing public pipeline API, tensor math, operation order, and model lifetime policy are unchanged. Split component loading preserves the former param-then-model order. Failed setup, model load, and extract boundaries are emitted where the detailed path observes them; completed denoise-step detail is retained if a later step operation fails. Child intervals describe composite host scopes: model load still includes ncnn read/unpack/pipeline creation/upload/waits, and extract still includes host dispatch plus internal waits. They are not pure disk or GPU-kernel timings.

## CPU verification

- Resource-bounded incremental build: `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 taskset -c 4,6 cmake --build build-dev --target ernie-image ernie-execution-metrics-contract -j2` — exit 0. Log: `artifacts/2026-09-06/o2-component-metrics/build.log`.
- `ctest --test-dir build-dev -R execution_metrics_cpu --output-on-failure` — 1/1 passed.
- Dedicated metrics CLI contracts rebuilt, then `ctest --test-dir build-o1/cli-stage-contract -R allocation_cli_cpu --output-on-failure` — 1/1 passed (13 cases inside the Python contract).
- `git diff --check` — clean.

The synthetic contracts verify complete and failed child records, coordinates, JSON serialization, and that child durations do not change top-level phase totals. They do not claim a real model failure was injected.

## Deferred actual run

A later fixed64 diagnostic run should use the already reviewed O1 input/model identities but freshly freeze this commit's source and ON binary. It should run metrics ON once, trace enabled, under the prior 10 GiB/swap-zero/two-CPU guard, then require the existing 27 trace tensor/PNG contract plus the new component coordinate/status schema. No GPU run was started in this slice. The observation will determine whether the repeated output-head model-load composite is large enough to justify a request-local cache experiment; this implementation does not enable caching or mapped loading.

## Independent-review fix

The first review found that explicit `Net` destruction happened while an `Extractor` (and on Vulkan, its command object) was still alive, and that text details were only flushed after all 25 blocks succeeded. The fix restores the required command → extractor → Net destruction order with inner scopes on every affected CPU/Vulkan path. Text setup, model-load, and extract failures now append a failed boundary, and the pipeline flushes completed plus current failed text details before rethrowing. The same CPU contracts and both affected targets were rebuilt and passed after the fix.

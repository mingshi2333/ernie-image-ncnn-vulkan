# O1 execution metrics bounded implementation report

## Implemented contract

The public `ernie/pipeline.h` API and default `generate` entry point are unchanged. A private `generate_with_metrics` overload accepts an explicit request-owned `ExecutionMetrics`; the normal entry calls the same implementation with a null collector. No process-global timing observer was added. The CLI selects the private entry only in an allocation-instrumented build when `--metrics-json` was actually provided.

The JSON allocation report now carries the execution snapshot. It records host intervals for verification, block model loading, block computation, inseparable DiT head/VAE/encoder load-and-compute, top-level upload/download, and the unaccounted per-step scheduler/packing remainder. Existing nested `DenoiseStepStats` totals are decomposed: block load and compute intervals are recorded directly, head intervals are recorded once as combined read/prepare, and only `step.elapsed - recorded children` is added for the remaining compute. The full step elapsed is never added again. `Net::load_model` is explicitly described as a combined host scope including reading, unpacking, pipeline creation/upload and internal waits; it is not called pure disk read. GPU durations are null because no timestamp query exists.

Submission counts use the already observed block/attention count and top-level initial/final transfers. Their JSON scope explicitly states that other internal submissions are unavailable. Upload/download bytes remain null rather than inferred from logical tensor sizes. Failure reports preserve any measurements reached before the primary failure and set `execution_finished_successfully=false`; formal speed and memory eligibility remain false.

## Validation

```text
# default instrumentation-OFF build
taskset -c 4,6 cmake --build build-dev --target ernie-image ernie-execution-metrics-contract -j2
exit 0
ctest --test-dir build-dev -R execution_metrics_cpu --output-on-failure
1/1 passed

# fresh isolated instrumentation-ON build; archived ON/OFF evidence was untouched
taskset -c 4,6 cmake --build build-o1/stage-on-v2 --target ernie-image -j2
exit 0

# fake generation and allocation lifecycle, both ON and OFF branches
ctest --test-dir build-o1/cli-stage-contract --output-on-failure
1/1 passed; 13 Python cases
```

The accounting test now proves submission-only observations do not invent upload/download bytes and rejects duplicate event identity. The CLI contract verifies host phase serialization, unavailable GPU time/transfer bytes, successful completion, and partial timing retention on generation failure. No model or GPU job ran.

## Remaining empirical work

This slice establishes the interfaces and bounded accounting behavior. Actual instrumented vs uninstrumented tensor equality, the 64x64 full generation, and the short/long/PE cost decompositions remain pending separately frozen runs and independent review. CPU RSS remains supplied by the parent process, and GPU time and complete transfer byte/submission coverage remain unavailable. Therefore the report cannot support a speedup or memory ratio claim yet.

## Failure-path review fix

Independent review found that completed denoise statistics were initially copied only after the entire denoise and final download returned. The mapping now runs in each post-step observer, immediately after `denoise` appends the completed `DenoiseStepStats`. If a later block, step, or final download fails, every earlier completed step remains in the report. The active failed step remains absent because its child intervals are not yet complete; the JSON continues to declare partial known interval coverage. A synthetic generation failure now records a completed compute interval before throwing and proves that both it and the prior verify interval survive in the failure report.

Both fresh ON/OFF CLIs were incrementally rebuilt after the fix, and the 13-case fake CLI contract passed. No model or GPU ran.

### Observer ordering closure

Independent review identified that completed-step metrics were still recorded after optional trace download/write. The CPU and Vulkan observers now record the completed `DenoiseStepStats` as their first action. Therefore a trace failure preserves the just-completed step; the currently executing step remains absent if denoise itself fails. The end-of-denoise aggregation remains removed, so successful steps are not counted twice.

CPU-only verification after this ordering change:

- `cmake --build build-dev --target ernie-execution-metrics-contract -j2`: pass
- `ctest --test-dir build-dev -R '^execution_metrics_cpu$' --output-on-failure`: 1/1 pass
- `cmake --build build-o1/cli-stage-contract --target allocation-cli-on allocation-cli-off -j2`: pass
- `ctest --test-dir build-o1/cli-stage-contract -R '^allocation_cli_cpu$' --output-on-failure`: 1/1 pass, 13 Python cases
- `cmake --build build-o1/stage-on-v2 --target ernie-image -j2`: pass
- `cmake --build build-o1/stage-off-v2 --target ernie-image -j2`: pass

No model or GPU execution was performed. The rebuilt binaries are not yet frozen as execution evidence; actual ON/OFF preparation remains gated on independent review closure and root authorization.

### Fixed 64 ON/OFF execution preparation

`outputs/execution-metrics-pipeline64-o1-v1` is prepared but has not run. It freezes the same 80-file tracked source snapshot for both sides, fresh ON/OFF binaries and their CMake caches, the historical fixed 64 prompt/latent/token inputs, and the existing 233-file model inventory. The authorized launcher authenticates the plan, worker, and supervisor before starting; the worker rehashes all frozen source files, inputs, the selected binary/cache, and live model files before invoking the CLI.

The supervisor contract is MemoryMax 10 GiB, MemorySwapMax 0, CPUs 4 and 6, a continuously sampled 3 GiB host-available floor, 1800 second timeout, and 50 ms sampling. The comparison requires exactly 27 trace files (25 FP32 tensors plus prompt and token IDs), no missing or unexpected records, byte equality for every trace file and PNG, and explicit diagnostic metrics checks. It requires successful partial-known-interval coverage, non-overlapping host interval scope, formal speed/memory eligibility false, required observed host phases, and null unavailable GPU time, CPU RSS, and transfer-byte totals.

Authorization hashes at preparation time:

- plan: `df0325ac89687b6f29065eaf43b19481be8e2567e71b2194d373c3c8509fc4c9`
- worker: `1bdee9fe128ea7da74d8fe8a22d383fbd95d31ab33af8928e729c482d8eea383`
- supervisor: `baba17d3b10c1582f3b9d38f9ab82b230c1b206b207e428c9bdbbcad9546f564`
- launcher: `9021309a41cc4fe8dcddba4293acae20ac287e7ddb1bd6c0946ac4edc29d8068`
- comparator: `472c9674e017edb92ae9aca8773653027123505d159e16d3a3fe2998da29a9d5`

The frozen status remains `prepared_not_executed`. No model or GPU process was started. Independent preparation review and root scheduling remain required before execution.

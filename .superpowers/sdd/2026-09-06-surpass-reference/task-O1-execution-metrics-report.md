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

# Task B3 report: paired end-to-end measurements

## Implemented

- Added `tools/port_metrics.py` with strict paired-record validation, case-level median aggregation, reference/candidate geomean ratio, unavailable-case evidence, fixed alternating AB/BA scheduling, protocol budget construction, and JSON serialization.
- A valid speed record now requires a matched case, input, saved-noise, model, precision, and PE setting; trace must be false; both ports must have successful runtime and quality status; both must expose matching external monotonic start/finish clocks and positive end-to-end wall time.
- Any failed, OOM, missing, traced, component-only, identity-mismatched, or quality-invalid pair makes the aggregate incomplete. The geomean and headline medians are withheld rather than calculated over survivors.
- Each case gets one new-process warmup per port followed by five new-process pairs ordered AB, BA, AB, BA, AB. Raw run entries are retained in the protocol schedule.
- Extended `tools/benchmark_pipeline.py` with BF16, prompt files, optional PE, opt-in trace, and optional img2img-facing arguments. Timing uses the host monotonic clock around the complete child process. Trace is disabled by default. The current native runtime has no PE-specific prompt-file or img2img input/strength CLI, so those requests produce a structured `incomplete` result before model verification; no capability is fabricated.
- Added focused rejection and scheduling/budget tests in `tests/test_port_metrics.py`, including a synthetic subprocess test for the unsupported img2img path.

## Validation

- `git ls-files 'tests/test_*.py' ... | xargs env PYTHONPATH=tools python -m unittest`: 66 tests passed.
- `PYTHONPATH=tools python -m unittest tests.test_port_metrics -v`: 9 tests passed.
- `python -m py_compile tools/benchmark_pipeline.py tools/port_metrics.py tests/test_port_metrics.py`: passed.
- `git diff --check`: passed.

Plain `unittest discover` also collected untracked tests owned by parallel B1/other work. One such test referenced an implementation not yet present in the shared worktree. It was excluded from the tracked 66-test baseline and is not a B3 regression.

## Coordination and schema

- B1 owns `outputs/port-corpus-v1/protocol.json`. B3 does not race that path; it exposes `build_protocol(...)` and `serialize_protocol(...)` for B1/root integration.
- B1 reported frozen cases with `id`, split, prompt/noise hashes, `[W,H]`, stage dtypes, model identity, and PE metadata. B3 consumes the corresponding normalized pair identity fields: `case_id`, `input_id`, `noise_id`, `model_id`, `precision`, and `pe_enabled`.
- B2 was given the strict per-side schema: `status`, `quality_status`, `wall_seconds`, `started_monotonic_ns`, `finished_monotonic_ns`, and `scope=end_to_end`, with pair-level `trace=false` and matched identity.

## Honest missing empirical inputs

No full model or GPU benchmark was run in B3. Therefore no `baseline.json`, speedup, hours, timeout, or disk requirement is claimed. `build_protocol` remains `incomplete` until root supplies measured 512, 1024, long-text, and PE calibration seconds, trace bytes per case, frozen formal cases, and currently available disk bytes. It then estimates the full fixed denominator without dropping failures. Formal trace archives still require per-case validation, packaging, and SHA256 recording by the integration/run stage.

The optional PE runner path is supported using the native CLI's existing greedy PE options. A separate PE prompt file is explicitly incomplete because the native CLI accepts one common prompt source. Optional img2img is also explicitly incomplete because the native CLI currently exposes neither input-image nor strength functionality.

## Review round 1 fixes

All four Important findings in `task-B3-review.md` were used as the implementation checklist.

- Frozen aggregation now requires exactly the ordered B1 performance cases `performance-0` through `performance-5`, with unique measured pair indices 0 through 4 for every case. It rejects absent records, duplicates, unexpected cases/indices, warmups, and incorrect AB/BA order. Protocol readiness also requires those six ordered cases, exactly five repeats, positive finite calibration values, and a positive measured disk value.
- Every side now supplies its own actual input ID, saved-noise SHA256 and `<f4` dtype, model ID, canonical weight SHA256 and proven status, PE identity, four-stage precision map, four-stage device map, trace state, runtime/quality status, scope, and clocks. Input, noise, model, weights, PE, and stage precision must match. Actual device maps are required but may differ between ports as permitted by the master plan.
- The benchmark snapshots the exact prompt and saved initial-noise bytes into the run directory before launch and hashes both. Formal eligibility requires `--latent` plus a matching frozen `--noise-sha256`; seed alone is ineligible. PE runs record the PE manifest hash and become ineligible when that package identity is unavailable.
- External timing is persisted in `finally`, including timeout paths. Results distinguish `timeout`, `crash`, `runtime_failure`, proven `resource_exhaustion`, and `resource_unknown`; return code and termination signal are retained. Unknown quality and failed quality remain aggregation failures.

Exact validation commands and outputs after the fixes:

```text
PYTHONPATH=tools python -m unittest tests.test_port_metrics -v
Ran 11 tests in 0.195s
OK

git ls-files 'tests/test_*.py' | sed 's#/#.#g;s#\.py$##' | xargs env PYTHONPATH=tools python -m unittest
Ran 95 tests in 0.705s
OK

python -m py_compile tools/benchmark_pipeline.py tools/port_metrics.py tests/test_port_metrics.py
exit 0

git diff --check
exit 0
```

The two subprocess failure tests are synthetic and CPU-only: one times out a sleeping Python child, and one terminates a Python child with SIGSEGV. No model or GPU job was run.

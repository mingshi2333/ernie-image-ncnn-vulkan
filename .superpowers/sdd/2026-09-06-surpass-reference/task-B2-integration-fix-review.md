# B2 integration fixes independent review

## Verdict

**Pass for the three Important findings in `task-B2-review.md`.** Commit `2579499` closes the PE-command and calibration-grid findings, and commit `df97a14` closes the unauthenticated/malformed-official-weight finding. I found no new Critical or Important issue within this review scope.

This verdict covers adapter/runner contracts and authenticated bounded weight auditing. It does not establish a baseline, full graph equivalence, quality acceptance, or a performance result. No model, GPU process, large-weight scan, or new weight hash computation was run.

## Original Important findings

### F1 candidate greedy PE command: closed

`tools/port_adapters.py:121-138` preserves the frozen greedy contract (`sampling=greedy`, semantic temperature 0) while emitting `--pe-greedy --pe-temperature 1.0` for the candidate. This is consistent with the current native implementation: `src/prompt_enhancer.cpp:26-29` validates a positive temperature before `sample_pe_token`, and `src/prompt_enhancer.cpp:44-45` returns argmax before any division by temperature. The reference command still receives temperature 0, which is its native greedy selector.

The frozen manifest contains exactly two enabled PE cases, `dev-pe-apple` and `performance-4`; both are greedy with temperature 0 and top-p 1. The adapter records PE precision/device separately and states that the candidate CLI temperature is unused for sampling (`tools/port_adapters.py:152-156`). The focused test also verifies that command construction does not mutate the frozen case.

### F2 all candidate calibration rows unavailable: closed

`tools/port_adapters.py:170-180` now applies thread constraints only to candidate rows and FP16/low-vram constraints only to reference rows. The resulting candidate four-thread FP32, FP16, and BF16 rows are `pending`; candidate eight-thread rows remain unavailable; reference FP16 and non-preflighted device-resident configurations remain unavailable. This is capability declaration only: pending does not mean quality or capacity passed.

### F3 unauthenticated or malformed official weights counted as matches: closed

`tools/audit_port_weights.py:77-118` requires a sidecar manifest for every selected official component, the exact pinned repository and revision, a lowercase SHA256, and equality with the complete safetensors file. It rejects duplicate JSON keys, bad header sizes, unsupported dtypes, invalid dimensions, malformed offsets, range/shape byte-count disagreement, overlap/gaps, and unreferenced payload bytes before emitting any official tensor row.

The original counterexample (`shape=[2]`, offsets `[0,0]`, absent provenance) now fails on missing provenance and also fails the byte-range check when provenance is supplied. Tests additionally cover wrong repository/revision/hash, negative and boolean dimensions, truncated/range-overrun payloads, overlapping ranges, and whole-file mutation.

The new MultiHeadAttention support is bounded rather than promoted to whole-graph proof. It derives all eight projection roles in ncnn load order, distinguishes non-square q/k/v/output shapes, accounts for FP16 weight padding followed by raw FP32 bias, and attaches a logical official name only when both the pinned param hash and pinned layer name match (`tools/audit_port_weights.py:121-132,163-196`). Logical acceptance then requires official component presence, exact logical shape, and canonical value hash (`tools/audit_port_weights.py:199-226`). Every audit remains `status=unproven`, `comparison_scope=product_comparison_only`, and `allowed_to_close_S=false`.

## Launch and input-evidence checks

`tools/compare_ports.py:44-60` copies the already hash-verified prompt, latent, and optional input image into the new run directory. The replaced adapter points command construction at that immutable copy, so the actual command cannot switch back to a subsequently changed corpus path. The synthetic mutation test confirms the launched program observes the original latent bytes.

Each launched side records complete text/DiT/VAE/scheduler precision and device dictionaries plus separate PE execution metadata (`tools/port_adapters.py:141-156`). These maps are configuration/source-contract evidence; they are not a new empirical observation of every runtime stage. Reference runtime log fields, when present, remain separately recorded under `log_observations`.

`tools/compare_ports.py:86-110` starts the monotonic interval immediately before opening/launching the timed subprocess and closes it in `finally`, including `Popen` failure. Timeout kills and reaps the process group and retains the wrapper return code. Ordinary positive exit 139 or 200 remains `failed` with no signal. A non-timeout child signal is classified as `crashed` only when GNU time writes its explicit `Command terminated by signal N` record. Failed, crashed, timed-out, input-mismatched, and quality-incomplete observations cannot form a speed ratio.

## Small-test and existing-artifact evidence

Executed:

```text
python3 -m unittest tests.test_port_adapters tests.test_compare_ports tests.test_weight_audit -v
Ran 23 tests in 0.237s — OK
```

This includes real tiny subprocesses for ordinary exit 139, SIGTERM through GNU time, timeout/process-group reap, and mutation of the live latent after snapshot verification. Weight tests use tiny generated files only.

The existing `weight-audit-vae-v3` JSON was read without rescanning weights. It reports 143 authenticated official tensors, 250 reference tensors, 141 content matches, 109 unmatched tensors, and `allowed_to_close_S=false`. All recorded official rows use revision `bc68c81e2a1730a394d5fc9fae70713dee940140` and have 64-character source hashes. Eight decoder attention projections have `logical_projection_value_match`; eight encoder projections remain `official_component_missing`, and the encoder scan records the unsupported Crop gap. The run provenance identifies the exact fixed generator hash, three VAE official components, a 900 MiB address-space limit, 50,400 KiB sampled peak RSS, and explicitly says `no inference`. These are historical recorded measurements, not rerun observations from this review.

The actual `calibration-incomplete-v4/result.json` in the main checkout has eight development cases. Both sides of every case are `incomplete` with `No executable/model configuration`; every pair contract is unproven/mismatched, `valid_pair_count=0`, ratios are null, report status is incomplete, and `superiority_claim=false`. Its copied manifest hash matches the report. This is a correct missing-configuration record and is not a baseline.

## Remaining limitations

- No configured candidate/reference development execution exists in `calibration-incomplete-v4`; fastest valid configuration selection remains pending.
- The calibration report currently feeds development rows to the performance-only 30-pair aggregator. Consequently its `metrics` section lists all six performance cases as missing and all eight development rows as unexpected. This is confusing suite diagnostics, but it remains fail-closed and does not create a ratio or baseline.
- Stage maps describe the requested and source-inspected execution contract. Except for optional parsed reference log fields, this review has no empirical per-stage runtime-attestation result.
- `weight-audit-vae-v3` is a VAE subset. Encoder attention official tensors are absent, the encoder scan stops at unsupported Crop, other reference components and derived constants remain incomplete, and graph gate S remains open.
- Full peer output, common quality verdicts, five-pair AB/BA performance measurements, memory measurements, and superiority gates remain unexecuted.

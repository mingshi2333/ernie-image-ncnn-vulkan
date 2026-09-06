# Q2 trajectory report independent review

Review scope: commit `6c50d7f`, `tools/diagnose_trajectory.py`, `tests/test_trajectory_report.py`, the BF16 extension in `tools/diagnose_pipeline_step.py`, committed `artifacts/2026-09-06/trajectory-text-conditioning`, and retained run metadata/tensor identities. No model or GPU process was run.

## Critical

None.

## Important

None.

## Verified execution order and completeness

`ordered_boundaries` describes the real sequence as text, padded text, three conditioning constants, initial latent, then prediction/euler-step pairs for every step, followed by final, unpacked and decoded (`tools/diagnose_trajectory.py:55-63`). `build_rows` rejects missing, duplicate and unexpected tensors and requires the reference to contain the same complete boundary set (`:66-78`). Each row binds its actual predecessor identities: DiT prediction uses the preceding latent, padded text, three constants and step schedule; each Euler output uses its prediction and preceding latent; final/unpack/decode each use their immediate predecessor (`:80-106`).

Both retained eight-step records contain 25 unique boundaries. The actual order is `text ... initial, prediction-0, step-0 ... prediction-7, step-7, final, unpacked, decoded`. The historical native-text run's first and only failed tensor boundary is `decoded`; the saved-official-text diagnostic has no failed tensor boundary. Thus the report does not select a later larger error in place of the first execution failure.

The artifact generator calls the existing independent auditor before constructing rows (`:111-127`). That auditor rechecks saved tensor and PNG metrics, hashes, runner/reference identity and unchanged gate files. I additionally matched both run result hashes, both reference fixture hashes, both runner hashes, every row candidate hash against the corresponding result comparison, the 25-element set/order, and the artifact's own script snapshot SHA256 `cbb5035d37ffcc803f69e171d293a07df2e81088964f7944c287827cd006fb9a`.

## Verified interpretation boundary

The two runs are clearly separated:

- Historical full path: `conditioning_source=native_text_encoder`, `native_acceptance_eligible=true`, 24/25 tensor boundaries, first failure `decoded`, overall `passed=false` although its separately reported PNG quantization gate passes.
- Injected official-text diagnostic: scope explicitly says native PE/text were bypassed, `conditioning_source=saved_reference_diagnostic`, `native_acceptance_eligible=false`, 25/25 tensor boundaries and overall diagnostic pass.

The README accurately states that official text removes the local decoded maximum failure while final/decoded NRMSE becomes slightly larger. It treats this as interaction sensitivity and does not claim that native text is accepted, that all error improves, or that text is the unique cause. The full native path remains the acceptance path.

## Fixed gates and BF16 option

The trajectory evidence uses the pre-existing saved FP32 gates; they were not fitted to either result. `diagnose_pipeline_step.py` adds `bf16` as an explicit CLI precision and gives it exactly the plan-required existing low-precision gate (`atol=.03`, `rtol=.25`, `nrmse=.15`) alongside unchanged FP32/FP16 gates (`tools/diagnose_pipeline_step.py:27,77-87`). No BF16 result is claimed by this artifact.

## Tests

```text
.venv/bin/python -m unittest tests.test_vae_cross_report tests.test_trajectory_report -v
Ran 10 tests in 0.003s
OK
```

## Assessment and remaining boundary

The saved trajectory report is internally consistent, bound to actual run identities, ordered by execution rather than error magnitude, and honest about the official-text bypass. It supports the next localized text/conditioning comparison but is one development fixture, not a formal result. Native 25/25 tensor acceptance, full PNG quality closure, broader long-text cases and any BF16 execution remain pending.

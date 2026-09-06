# B2 adapter and measurement integration fixes

The two adapter Important findings in task-B2-review.md have been corrected. The independent weight-audit fix is df97a14; complete B2 baseline acceptance remains pending actual peer model execution and graph/weight correspondence.

- Frozen greedy temperature=0 retains its meaning. The candidate receives `--pe-greedy --pe-temperature 1.0`, because its positive-temperature validator runs before greedy selection; the reported CLI value is explicitly unused for sampling. The pinned peer still receives temperature=0.
- Calibration eligibility is scoped to each port. Candidate FP32/FP16/BF16 with four threads remain pending; peer-only FP16/capacity restrictions no longer reject every candidate configuration.
- Stage dictionaries contain text/DiT/scheduler/VAE, with PE precision/device/sampling metadata recorded separately.
- `compare_ports` passes both frozen manifest and protocol to `summarize_pairs`. Each observed side carries the canonical full case, model, shape order and image-input identities required by the metrics contract.
- Verified input bytes are copied into each run and those copies are actually passed to the native command. Subsequent mutation of the source noise does not change the executed noise.
- Launch attempts retain complete monotonic clocks even when `Popen` fails. Timeouts kill and reap their process group and retain the exit status. Ordinary positive exit 139 remains a failure; a child signal is recognized only from the GNU time wrapper's explicit signal report.

Validation: 15 adapter/launch tests pass, including actual tiny child processes for positive exit 139, SIGTERM, timeout, and source mutation after verification. The launch-error path is injected before process creation. These are protocol tests without models; a successful tiny process never receives a quality pass.

Actual calibration-incomplete-v4 contains all eight development observations as missing executable configuration, no ratio and no superiority claim. This verifies the current manifest/protocol integration failure path, not a model calibration.

Still incomplete: authentic full peer output, common weight/graph identity, independent full quality verdicts, five paired AB/BA measurements, and fastest valid configuration selection. This change intentionally retains `quality_status=incomplete`, `weight_identity_status=unproven`, and the absence of performance ratios until those required observations exist.

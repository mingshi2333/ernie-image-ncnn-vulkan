# Q1 diagnostic fix independent rereview

Review scope: commit `8e4430f`, the two Important findings in `task-Q1-review.md`, `tools/diagnose_vae_cross.py`, its tests, committed `artifacts/2026-09-06/vae-cross-v2`, and the retained four-cell files under `outputs/vae-cross-pe-v2`. No model or GPU process was run.

## Critical

None.

## Important

None. Both previous Important findings are closed.

### Closed: official workers execute the frozen dependency snapshot

The driver copies every local Python tool into one snapshot and records each pre/post-copy SHA256 (`tools/diagnose_vae_cross.py:35-50`). Before and after every child it verifies every saved dependency (`:53-56,236-238`). Both official cells execute `snapshots/diagnose_vae_cross.py`; imports therefore resolve from that snapshot directory, while the explicit `--source-root` is used only to point `prepare_block.ROOT` at the verified read-only configuration/weight assets (`:127-151,230-235`). Native cells execute the copied runner. The four cells remain sequential (`:216-240`).

Actual evidence agrees with this design. All 56 files in `outputs/vae-cross-pe-v2/snapshots` match `worker_sources`. The executed and committed snapshot hash is `564c56f0492f5a9d319e709c8b8d635e5438ec7e92bb53bbd9759400c5b7d92a`. Both official logs report that worker hash, the same official class hash, the same decoder/post-quant hashes, CPU and FP32. The actual commands point at the snapshot rather than the live tool.

### Closed: bitwise reproduction compares dtype, shape, and bytes

`bitwise_equal` now requires identical shape and dtype and compares C-order bytes (`tools/diagnose_vae_cross.py:25-29`). The regression test explicitly distinguishes positive and negative zero, dtype changes and shape changes. The retained results report both historical reproductions as bitwise equal; independent SHA256 reads confirm the historical official/`oo` file hash `2e3df056...52e76db9` and historical native/`nn` hash `70f3ecd3...08e` are exact matches.

## Recomputed evidence

- Artifact `results.json` is JSON-identical to the retained output result.
- All four actual output files match their recorded SHA256. `oo/no` share official-latent hash `e846dc43...11765`; `on/nn` share native-latent hash `24b02522...e7e53ce`.
- Re-reading the four little-endian FP32 arrays as `[1,3,384,512]` reproduces:
  - decoder on official latent: max `7.152557373046875e-6`, NRMSE `2.5668685555158667e-7`, pass;
  - decoder on native latent: max `2.9206275939941406e-6`, NRMSE `2.5291233967463773e-7`, pass;
  - native versus official latent through official decoder: max `0.011106044054031372`, NRMSE `0.00010028426399443022`, fail on the fixed local maximum gate;
  - connected native/native versus official/official: max `0.011105477809906006`, NRMSE `0.00010027834945528683`, fail on the same fixed gate.
- These values support continuing upstream latent diagnosis. They do not assign a causal percentage and do not prove full image acceptance.

## Tests

```text
.venv/bin/python -m unittest tests.test_vae_cross_report -v
Ran 6 tests
OK
```

## Assessment

The review repair is effective and the rerun evidence supports the stated Q1 diagnostic conclusion. Q1's complete 25/25 tensor and PNG acceptance remains open, exactly as the artifact README states.

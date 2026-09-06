# Q2 next bounded diagnosis — CPU evidence

2026-09-06. No new GPU execution, runtime change, precision grid, default change or acceptance-gate change. Companion `task-Q2-next-evidence.json` records actual hashes and independently recomputed metrics; scratch scripts `q2_next_cpu.py` and `q2_next_supplement.py` reproduce this analysis with two CPU threads.

## Earliest supported boundary

The first confirmed divergence is already prediction **0**, with initial latent bitwise equal. In the historical official-text full trajectory, padded text is also bitwise official, yet prediction 0 has NRMSE **8.5102094e-5**, max **0.0033519268**, L2 **0.084575334**. After the first Euler update, latent error L2 is **0.0029165101**. Therefore native text error is not a necessary cause of all accumulated latent error. This does not locate a specific DiT operator: that call still includes timestep-feature construction, input heads, block stack, and output head.

For the first update, FP32 sigma delta is -0.03448277711868286. The contribution `delta * (native_prediction - official_prediction)` has L2 0.0029163924; the residual of the measured update difference after subtracting this contribution is L2 2.5057247e-5, max 4.4395182e-7. This bounds the directly observed update-arithmetic discrepancy for this boundary. It does not establish that every scheduler/timestep operation or later step is irrelevant.

| Saved trajectory | Prediction 0 NRMSE | Prediction 0 max | Latent after update 0 L2 |
|---|---:|---:|---:|
| pre-compensation FP32 | 1.6100875e-4 | .0066553354 | .0055177903 |
| Kahan / chunked FP32 | 1.0894244e-4 | .0048897266 | .0037334752 |
| official-text FP32 | 8.5102094e-5 | .0033519268 | .0029165101 |
| saved-vector-text FP32 | 1.3440982e-4 | .0074050426 | .0046061933 |

The chunked and official-text trajectories use exactly the same binary SHA `1dd2880274fd862631254966f78e401340393a711be7b27940554715fe13320b`. Their text intervention is particularly interpretable. Other rows use different binary identities; numerical changes must not be attributed to only one operator without the corresponding source/option evidence.

## Prompt, token, bucket, execution identity audit

All five trajectories in the table were checked against the actual command literal or actual `--prompt-file`, native `trace/ids.txt`, full official fixture, and official text fixture. Prompt is `雪山脚下的蓝色湖泊，松树林，清晨阳光，写实风景摄影。`, **78 UTF-8 bytes, no trailing LF**, SHA `d778f6ec02a9bfdf47dcaa14b869574a110876e9cca248678235f440f23c18f1`. All have the same 32 token IDs, ending **1320**. Selected text bucket and DiT text length are both **64**, not 32; 25 text blocks and 36 DiT blocks. The full reference fixture SHA is `81a853d9db2c6aba80a3fa72abac618dd9196f8598b5055cc2a41586480580b7`; initial latent SHA `f07a2134e662229cd4f6981c648cbbc925080debf514ef51c00d7669dc55e983` is exact in every run. Package identity is the previously fixed `72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1`.

Actual runner and validator copies were rehashed, as were every recorded Python source snapshot file and all eight native/reference prediction and latent tensors. Each run records Vulkan DiT FP32 and FP32 text/scheduler/VAE; actual commands request FP32/direct VAE convolution. The common reference resides beneath a historical directory whose name contains `fp16`; its executed oracle configuration is FP32, so the directory name is not precision evidence. Historical image-run snapshots do **not** contain a complete native build-source manifest; this limitation is explicit in the JSON. Do not retroactively represent a later complete source inventory as the exact build provenance of those binaries.

The saved-vector full run explicitly uses `--embeddings`; it is a diagnostic embedding intervention, not a native end-to-end text execution. Its `trace/text.f32` matches that supplied file. The upstream candidate has SHA `c9eef032e54fe6a2a0d3eeb13458f66821a3442308684b330fa2ca72032c0138`. In `q2-history-vector-down-v2/identity.json`, `worker` names the frozen `diagnose_text_stages.py` (SHA `59c84755eff3945bdf4e0c92cae6e7d10401508680b33baca01c397d90b7157e`), **not** `run.py.snapshot`. The launcher is separately SHA `a09804106e1b1e3dc3fa53c0dd6bb8ffdf30d34b0c2c0e55bba8bf020ea39bc6`. Its source inventory explicitly says it was observed after process start; it is not a compilation input manifest. The initial supplemental check compared the wrong two roles and stopped at an assertion; the corrected script checks the named worker. This was an analysis setup error, not artifact corruption.

## What the completed step-6 interventions establish

The current `a4a80b...` block-sequence binary reproduces the original native prediction 6 **bitwise** from the six saved native inputs. Against official prediction 6, this replay has NRMSE .00259902085 / max .0997596383. With native constants and native timestep features fixed:

- Replacing only incoming latent with official step-5 latent gives NRMSE **1.6918865e-5**, max **.0005716085** against official prediction 6. Its absolute change from replay has L2 **2.15618856**.
- Replacing only padded text gives NRMSE **.0025987740**, max **.0997976065** against official prediction 6. Its change from replay has L2 **.0056270134**.

These are single-boundary sensitivities, not a causal allocation of the entire trajectory. They show that accumulated incoming latent divergence is the distinguishing local input at step 6; they cannot identify where that incoming divergence was created. An official-input teacher call has NRMSE 1.2358507e-5. Swapping only its timestep feature to native gives 1.0669699e-5; this fails to support native timestep feature alone as the cause of the observed step-6 failure. All comparisons retain their distinct labels and `native_acceptance_eligible=false`.

## Existing layer evidence and its limitation

The saved step-0 layer and exact-head experiments use **old runner `bf5b7f26344d5faa8c46316c988596637793708e862b5c137642d2d1852aeaef`**, not current a4a. Actual runner, fixtures, selected layer tensors and exact-head ten inputs were rehashed. Whole-hidden-state NRMSE grows from 1.31624e-6 at block 0 to 2.75060e-5 at block 23, 1.05166e-4 at block 29 and 1.68955e-4 at block 35. The experiment supplying exact official projected hidden state and all six modulation tensors still ends with hidden NRMSE **1.5251538e-4**, max **4.1365967**, L2 **220.474942**.

That proves input-head error was not the sole source of the **old** stack's hidden-state divergence. It does not establish the current compensated stack's profile, and differences between successive layers include propagation of already different inputs. They are not isolated operator error measurements. Existing corrected real-QKV SDPA evidence documents compensation improvements; the earlier malformed MemoryData fixture is invalid evidence. No basis presently supports changing Gemm, residual, timestep, or attention defaults from these historical boundaries alone.

## One next experiment and falsification conditions

Root has accepted preparation of exactly one current-a4a **step-0 exact-head block-stack replay**: ten existing official inputs, 36 blocks, FP32 Vulkan stream, 4160 tokens, all block traces. No input/output heads, scheduler, text encoder, or VAE. It reuses existing official per-block tensors; no official model execution or precision grid. Preparation must bind all ten input bytes to the original head/stage fixtures, layer numbering/axes and all 36 denominator tensor hashes, actual runner and frozen source, and guard resources. GPU execution remains queued behind O1 and requires explicit release.

If current exact-head drift persists, the input-head and outer Euler arithmetic cannot be the sole source of this block-state drift. The first layer with a reproducible increase then determines a **same-input** local-operator diagnostic; the profile itself does not justify a Gemm/attention accusation. If current exact-head drift is much smaller than the old profile, reject transplanting the old layer-localization and examine the current head/constant boundary next. This comparison has hidden-state outputs, not a prediction/final-image gate, and cannot by itself apportion current full-run prediction error or repair the free-running failure.

## Prepared exact-head execution (not run)

`outputs/q2-chinese-step0-current-exact-head-stack-v1/plan.json` freezes ten copied inputs, all 36 existing oracle tensor hashes/shapes, 212 predictor snapshot files, runner a4a, and 72 fixed-manifest model files. CPU preparation rehashed all input and oracle tensor bytes and checked finite FP32 values. The 72 model files have fixed expected hashes and size checks now; their actual full bytes are explicitly rehashed immediately before execution. This stage does not falsely report that preparation already rescanned all model bytes.

`tools/diagnose_exact_head_stack.py` is copied into the execution directory and imports only standard library/numpy. The worker executes that frozen copy, not live `ROOT/tools`. The historical official fixture's saved generator SHA was rechecked (`5440e1...`), as was its entire recorded Python snapshot. That historical launcher executed its original `__file__`, and a copied snapshot is not a complete historical dynamic-import log; `execution/oracle-provenance.json` preserves this limit. Head/source/axes mapping is validated against the fixed whole-fixture hashes: in0=head-0; in1..6=head-2..7; in7..9=original in3..5. Head-1 is the output-head timestep embedding and is intentionally unused. Official post-block outputs transpose from sequence-major to [1,4160,4096]; native plain packing download writes [4160,4096] in the same contiguous order, including all image and text tokens.

Predeclared comparison denominator: exactly block-0 through block-35, all 4160×4096 values each, FP64 sums for NRMSE/L2/max, and final raw output must equal block-35 bytes. Missing/extra traces, sizes, hashes, NaN/Inf and changed command/source fail closed. Five synthetic tests pass (complete contract, missing layer, wrong modulation, transposed axes, finite/numeric metrics). Initial test collection caught a missing Python list bracket before preparation; corrected and rerun 5/5. No model/GPU job was started.

Queued command, only after root GPU release:

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 0,2 .venv/bin/python outputs/q2-chinese-step0-current-exact-head-stack-v1/execution/worker-v2.py
```

Guard uses recursive descendants including separate sessions, sampled summed RSS 9 GiB, entire GPU 6144 MiB, host available ≥3 GiB, 2400 s deadline and 0.5 s samples. GPU-monitor failure/timeout kills the tracked process tree. Native frozen binary requests four threads but affinity restricts it to two physical CPU cores; this is disclosed, not mislabeled as a native two-thread option. Planned traces alone are 2,453,667,840 bytes; runtime requires an extra 1 GiB free disk. `worker.py` is retained unexecuted; `worker-v2.py` adds fail-closed GPU monitor timeout/OSError handling. Execution success is labeled process completion only, never a numerical gate.

Plan SHA `3aba6aa5ec1bec491a36187bc0f62c5332549d26df396655010d1966db4201f7`; frozen diagnostic SHA `f98954eaa224c8d023dbb35046469cca8c4c584aee96ebb3688cc33afc2e550a`. `execution/identity-v2.json` binds the final worker and provenance. GPU remains unused and queued behind O1.

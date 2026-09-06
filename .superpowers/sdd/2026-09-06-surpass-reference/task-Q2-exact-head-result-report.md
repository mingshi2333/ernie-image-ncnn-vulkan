# Q2 current exact-head 36-block result

2026-09-06. One authorized execution, session 94495, completed exit 0. GPU released as soon as the native child exited; subsequent audit is CPU only. This is **descriptive hidden-state evidence**, not official prediction/image acceptance, and does not repair the Chinese free-running failure. No runtime/default/gate change.

## Execution and identity

Frozen worker-v2 SHA `a7c50ae78907cbbf7f961ac738314df77669067f3c477583ae1b09fd3df5d424`, plan SHA `3aba6aa5ec1bec491a36187bc0f62c5332549d26df396655010d1966db4201f7`, predictor SHA `a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc`. Execution used exact official step-0 projected hidden state, six modulation tensors and official cos/sin/mask, 4160 tokens, 36 ordered FP32 Vulkan blocks with stream policy. No input/output heads, Euler scheduler or VAE. Prompt/IDs/64-bucket lineage and historical oracle-source limitations remain as documented in `task-Q2-next-report.md` and frozen `execution/oracle-provenance.json`.

All 72 actual model files passed full-byte SHA checks before native startup. After execution, independent CPU audit rehashed them again, together with every bound source/binary, all ten input copies and original inputs, all 36 oracle and native traces, final output and plan. Final output is bitwise block-35 trace. A separate FP64 reduction using different chunk sizes/dot products reproduced all 36 whole-tensor metrics within 1e-12. All values finite; complete denominator **36 × 4160 × 4096**, no missing/extra traces. Existing five synthetic tests remain the preparation evidence; no repeated GPU calls were added.

Worker elapsed **107.272980147 s**, return code 0, no stop reason. Sampled recursive RSS peak **1,012,318,208 bytes**, entire GPU peak **3912 MiB**, minimum host available **17,511,260,160 bytes**. Limits stayed at 9 GiB RSS / 6144 MiB GPU / 3 GiB host / 2400 s; physical-core affinity 0,2, with the frozen native binary's internal four-thread request explicitly preserved. `passed=true` in worker means execution completed, not a numerical gate.

Result SHA `bc8ddc450ebce74400c3b2f3ec4c1a6e748f5323ceeb314521aae65e54089b63`; worker-result SHA `74e7ed2535df98542b9a4a7e683efadeb4f62a3056dc64681b41a97fe499d54a`; final FP32 SHA `dfc42879828b9d635c8a0b170e36b6d7b198e7f437c7a14d450bc24829a23c49`. Independent reduction and resource details are in `task-Q2-exact-head-result.json`; reproduction script is `q2-review-scratch/audit_exact_head_result.py`.

## Earliest observed difference and growth

Exact official input first becomes observably different at **block 0**, NRMSE 6.3652792e-7, L2 .111847604, max .00390625. Thus there is now current-run evidence for rounding divergence inside the block stack, independently of text encoder, timestep/input-head arithmetic and outer Euler updates. This statement is about block hidden state, not a quantitative attribution of full-run final-image error.

The first twelve layers stay near 6–8e-7 NRMSE. A sustained rise appears across layers 13–17 and continues later. This is an observed profile, not a predeclared statistical change-point or evidence that block 13 introduced a new implementation defect. The L2 norm and reference scale must be read together: block 35 NRMSE decreases from block 34 while absolute error L2 increases.

| Post-block index | Current NRMSE | Error L2 | Max abs | Old exact-head NRMSE |
|---:|---:|---:|---:|---:|
| 0 | 6.3652792e-7 | .111847604 | .00390625 | 1.2528447e-6 |
| 12 | 8.3457441e-7 | .426644435 | .01612091 | 1.3992491e-6 |
| 13 | 1.0423372e-6 | .485453638 | .01753235 | 1.7749533e-6 |
| 14 | 1.3361513e-6 | .537792979 | .01602173 | 2.2436884e-6 |
| 15 | 2.0356308e-6 | .673116976 | .02880859 | 3.4036439e-6 |
| 16 | 2.6368741e-6 | .923859888 | .05224609 | 4.7877948e-6 |
| 17 | 3.5660106e-6 | 1.20791946 | .09008789 | 6.7981110e-6 |
| 28 | 3.0310217e-5 | 7.18534822 | .58666992 | 6.8979840e-5 |
| 29 | 4.1877525e-5 | 10.3736494 | .70593262 | 9.2699279e-5 |
| 34 | 9.3608455e-5 | 67.8168241 | 1.24462891 | 2.0460895e-4 |
| 35 | 6.9826350e-5 | 100.940377 | 1.51690674 | 1.5251538e-4 |

The old column here is specifically **old exact-head** runner bf5b, not old full-head traces. Current stack error is smaller, but old-vs-current is a different-binary comparison, not a uniquely isolated attention patch effect. The current experiment supplies the missing evidence that accumulated block-state divergence persists after the compensated implementation; it does not simply transplant the old result.

Whole-tensor norms are dominated by image tokens partly because there are 4096 image tokens versus 64 text tokens. Separate normalized measures show an additional difference: at block 15 image-token NRMSE is 2.1173763e-6, text-token NRMSE 6.2060483e-7; at block 35 they are 7.0362776e-5 and 1.8129193e-5 respectively. The final image-token error L2 is 100.885012 versus text-token 3.342790. These are within-DiT hidden-token groups, not text-encoder error measurements.

## Next bounded diagnosis, not an immediate runtime candidate

Do **not** change Gemm, SDPA, residual or timestep defaults from this profile. Each post-block comparison mixes local arithmetic error with propagation of previously different input. A layer-to-layer error increment is not that block's standalone rounding error.

Recommended next target is **block 15 with exact official block-14 hidden state**, the same six modulation/cos/sin/mask inputs and current options. This chooses an early part of the sustained rise while its absolute error is still small. First use the already saved official block-15 output as the complete denominator; if adding internal probes, keep them within this one block and authenticate their graph outputs against the fixed param/source. The concrete existing graph boundaries are:

- modulated prenorm → Q/K/V Gemm outputs 19/22/25;
- post-normalization/RoPE Q/K and V → blobs 55/69/27;
- compensated SDPA output 70, attention projection 73, first residual 75;
- MLP prenorm/modulation 81, Gemm branches 84/85, gated hidden 87, down projection 88 and final `out0`.

A same-input block-15 output is the prerequisite; internal distances after a diverged preceding operator still do not alone prove that downstream operator is inaccurate. If exact-input block-15 error is much smaller than the observed full-stack block-15 L2 .673117, that falsifies a claim that local block-15 arithmetic alone explains the current boundary gap and favors propagation/sensitivity analysis. If it is comparable, that motivates the above within-block boundaries, not an automatic precision change. A specific operator becomes actionable only after testing it with identical saved inputs on both sides. Any diagnostic extraction must also reproduce the uninstrumented block output so observation effects do not masquerade as a fix.

No next experiment was executed or broad grid prepared in this task. GPU is free; formal acceptance remains pending.

# Q2 block-15 official-input teacher result

One authorized small GPU call completed; GPU released and both other agents confirmed no overlapping GPU work. Session 92666, 5.388071801 seconds, exit 0. No production/default/quality-gate changes. Commit contains diagnostic tooling, four passing synthetic tests and CPU audit/report only.

## Identity and complete denominator

Used the **identical a4a block-sequence binary** from the successful 36-block experiment, with only one `--model .../dit/block-15`; no replacement executable or rebuilt math. Source `run_block_sequence` iterates the supplied model list and applies each actual model. Local observer name `block-0` therefore means **global model block 15**, not model 0. No heads, scheduler or VAE. Conditions in1–in9 are byte-for-byte the prior exact-head conditioning. Input in0 is existing official post-block-14 hidden state, fully checked against the fixed earlier fixture and denominator. Output is complete contiguous FP32 [4160,4096], **17,039,360 values**, with final output bitwise equal to the sole local trace.

Frozen plan SHA `7c28ba7632bd4f2a7a26c2c9d00b61bc65c7ddf1eb9a6310c0bd90c1d8a8bfe3`; worker SHA `e734ac72fa540d78015ff0ebc5188fff8bab85ae3f684ba983167ab20ac99f30`; runner SHA `a4a80b564042d4020096edf947d30bd28a5fdf354005d1c17685fddddcdbe1fc`. All source/model/input/reference identities were checked before execution and independently rehashed after it. Both block-15 actual model files match the previously fixed package manifest. The worker runs the frozen diagnostic and helper copies, not live tools.

Actual output SHA `175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3`; result SHA `03b41a93fcd282c45e224dafecc5f14f72362ae3430cb92b825169dbebe42867`. Full input identities are recorded in `task-Q2-block15-teacher-evidence.json` and output `plan.json`.

Guard retained 9 GiB recursive RSS, 6144 MiB whole-GPU, host available at least 3 GiB, timeout 2400 s and physical-core affinity 0,2. Native internal thread request remains four. Sampled peak RSS **846,036,992 bytes**, whole GPU **3913 MiB**, minimum host available **17,257,680,896 bytes**. No stop reason. After process exit, GPU returned to 1635 MiB. Process success is not a numerical acceptance flag.

## Local versus incoming discrepancy

| Complete output comparison | NRMSE | Error/difference L2 | Max abs |
|---|---:|---:|---:|
| Native block 15 on official input vs official block 15 | 6.011371945e-7 | .198776545 | .006591797 |
| Accumulated native stack block 15 vs official block 15 | 2.035630767e-6 | .673116976 | .028808594 |
| Accumulated native stack block 15 vs teacher native block 15 | 2.046408418e-6 | .676680984 | .028076172 |

The first row is a same-input implementation difference. The third is a conditional native response to changed incoming hidden state; it must not be called pure rounding error. Prior native block-14 input differs from official input by L2 **.537792979**. Its downstream block-15 change has L2 **.676680984**, a conditional directional L2 gain of **1.25825552**. This is a measured gain for this saved perturbation, not a global condition number or Lipschitz bound.

Independent CPU vector decomposition used `local = N(official14)-official15` and `propagated = N(native14)-N(official14)`. Their sum is exactly the algebraic total native-versus-official output difference. Their inner product is **-.0221614028**, cosine **-.1647586171**: they partially cancel. Norms are not additive, so no causal percentage is claimed. The complete metrics were independently reproduced using different chunking and dot reductions; finite checks and byte hashes passed.

This rules out **block-15 local same-input arithmetic alone** as an explanation of the accumulated .673117 output gap. Existing incoming hidden divergence has a larger measured effect at this boundary. It still does not prove whether that divergence originated in attention, projections, residual arithmetic, or MLP elsewhere, nor does it show that more precise arithmetic would reduce final image error.

## Exactly one next internal boundary

Choose **first residual output blob 75 (`ErnieResidualAdd add_16`)** of this fixed block-15 graph, rather than dumping every blob. It separates the attention/modulation/projection/residual segment from the MLP/modulation/residual segment. Use the same two saved incoming hidden states (official14 and native14) with all nine conditions unchanged, and compare their native blob-75 difference. No new full model trajectory or precision grid is needed.

If most of the measured amplification from incoming L2 .537793 to final difference .676681 is already present at blob 75, prioritize a later same-input diagnosis inside the attention-side segment. If it appears predominantly after 75, prioritize the MLP-side segment. This branch-level sensitivity is not a proof that SDPA or Gemm itself is inaccurate. Blob 75 has the same [4160,4096] state axes, so the state-difference norms are directly interpretable, with cancellation/nonlinear propagation still explicitly possible.

Before using an instrumented extractor, require its final `out0` to reproduce the two already saved uninstrumented outputs bitwise. Extracting a retained intermediate can alter memory/pipeline behavior; a mismatch invalidates the instrumentation comparison and must be investigated first. Any future official/internal operator comparison needs actual matching input bytes and independently generated official boundary evidence. No next internal extraction was executed in this task; current result already selects its single most useful boundary without changing runtime math.

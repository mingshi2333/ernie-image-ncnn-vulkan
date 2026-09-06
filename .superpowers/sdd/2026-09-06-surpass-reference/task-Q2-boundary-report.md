# Q2 block-15 internal boundaries and concrete down-Gemm hypothesis

Scope: authorized bounded continuation of the same block-15 diagnostic. **No production math, default, CMake or formal gates changed.** Three selected boundary pairs were executed, each only after the previous result selected the next useful boundary. GPU is released. We now have a testable operator candidate; this report stops before implementing a changed-math candidate.

## Instrumentation validity

A standalone diagnostic C++ executable reads the existing graph and weights, uses the same four-thread FP32 Vulkan device-I/O/session allocator/shared pipeline-cache options, retains one intermediate, then extracts complete out0. It is compiled independently outside build-dev using the five custom-layer .cpp/.h files copied from the a4a snapshot. Their bytes match a4a. The existing ncnn static library and generated SDPA header are frozen; no ncnn math source was changed. This is a new executable, not a claim of binary identity with a4a. Exact endpoint checks provide the additional necessary per-fixture evidence.

Every instrumented out0, **both inputs at all three boundaries**, exactly matches the corresponding uninstrumented output: official-input teacher SHA `175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3`; accumulated-native-input SHA `55c6e86b1bc14acb1acdcd15383c1055ee8fe46ff744657418435370fdadc5e6`. Each check covers all 17,039,360 FP32 values. Thus retained-intermediate observation did not change these measured endpoint bytes. This is not an unconditional proof about all graphs or inputs.

The pair tool refuses boundary comparisons if the first final hash differs, and stops before the second input. Two synthetic negative tests verify this ordering and rejection of an unreviewed boundary; **2/2 pass**. It allows only the specifically reviewed boundaries 75, 88 and 87. The C++ probe currently has additional dormant boundary names, but the frozen Python protocol does not authorize their execution.

Initial direct build attempts failed on missing relative ncnn include roots and then missing generated SDPA header. Correct absolute include roots and a frozen generated header resolved these setup failures. Build source and library copies were retained. The first CPU reconstruction attempt failed before creating output because SciPy was unavailable; it was replaced with standard-library `math.erf`, and that initial failure log is retained. No install or model rerun was needed.

## Three bounded pairs

| Boundary | Reason selected | Whole boundary denominator | Session | Wall s | Sampled peak RSS bytes | Whole GPU MiB |
|---|---|---:|---:|---:|---:|---:|
| 75, first attention residual | Separate attention-side from MLP-side propagation | 4160×4096 | 63671 | 11.2612 | 899448832 | 3905 |
| 88, MLP down projection | Reconstruct final gate/residual and test its arithmetic | 4160×4096 | 41714 | 10.1074 | 976994304 | 3913 |
| 87, actual input to down Gemm | Isolate Gemm local error with identical actual inputs | 4160×12288 | 3102 | 10.6870 | 1079209984 | 3948 |

All exit 0, no resource stop. Guards retained 9 GiB recursive RSS / 6144 MiB entire GPU / 3 GiB minimum available host / 2400 s and core affinity 0,2. Intermediate 87 uses 51,118,080 FP32 values, not the smaller state denominator. GPU was released and announced after each pair.

Directories are `outputs/q2-block15-boundary-v1`, `...boundary88-v1`, and `...boundary87-v1`. The first two use runner SHA `fa6fad2b5433a75ddabe046c64cbfd382096817d65efe4b200ae901e6ca5f9a5`; 87 uses `ec43902d86e5036dadd2ac308a3445355407097bb79023a088520a76dbeca0ca` with only the accepted-name list expanded. Frozen pair scripts, worker scripts, plans, custom sources, static library and checksums remain in their execution directories. Actual bound files and all six endpoint/boundary identities were independently rehashed in the CPU audit scripts.

## Propagation and final residual: negative result for a residual bug

Conditional L2 differences for identical conditioning but official14/native14 hidden inputs:

- input hidden: **.537792979**;
- first residual 75: **.568827585**;
- down projection 88: **.236487279**;
- gated MLP branch `in6 * 88`: **.542190548**;
- final hidden: **.676680984**.

The learned modulation gate ranges from **-22.4982643 to 19.6026173**. For both inputs, CPU `FP32(blob75 + FP32(in6 * blob88))` reproduces **every final FP32 value bitwise**. The difference between the two residual-rounding terms has L2 **.0113032575**. This fixture does not support a final residual shader deviating from standard FP32 arithmetic. Changing residual math first would lack this evidence.

The pre-residual perturbation and gated branch perturbation have cosine **-.258982041**, so their effects partly cancel. Norm differences are not causal percentages. The larger late growth points toward investigating the MLP branch, but does not on its own distinguish mathematical sensitivity from floating-point implementation error.

## Small-token mathematical oracle, with explicit limits

The next CPU calculation selects **five tokens before recomputation**: image/text endpoints 0/4095/4096/4159 plus token 22, the largest already-saved gate-weighted down-output perturbation. It evaluates the actual MLP equations on both saved blob-75 inputs using official block-15 BF16 weights expanded exactly, RMSNorm epsilon 1e-6, fixed shift/scale, gated GELU MLP and down projection. Weight component SHA is `d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2`, authenticated by fixed package source_weights[15]. Reviewed diffusers source SHA is `0f1814f63008298707afea5d7bd22d0a16073f0a8e858037c198faa28027e926`; source defines `linear_fc2(up_proj(x) * gelu(gate_proj(x)))`.

This is an independent **FP64 mathematical reconstruction** and CPU FP32 comparison, not execution of the official CUDA module, and not a complete 4160-token oracle. It uses standard-library erf and NumPy BLAS. Source/weights/selected rows are explicit in results; no quality thresholds are inferred from the selection. CPU v2 completed in **5.61 seconds**, measured maximum RSS **869952 KiB**, two-core affinity and two BLAS threads.

Selected rows × both inputs:

| Quantity | L2 | Max abs |
|---|---:|---:|
| native MLP 88 minus FP64 MLP | .00487596659 | .000259598777 |
| CPU FP32 MLP minus FP64 MLP | .00105992110 | .0000344557941 |
| native conditional output change | .0271133446 | .00210952759 |
| FP64 conditional output change | .0267245809 | .00202307019 |
| native change minus FP64 change | .00348753025 | .000282515272 |

Much of the measured sensitivity remains in high-precision mathematics. Native nonetheless adds more rounding difference than this CPU FP32 reconstruction. That comparison alone does not identify an operator.

A separate FP64 evaluation of the native GELU polynomial isolates its intrinsic approximation error: L2 **.000136059650**, max **.000004097940**. Its cosine with the total native MLP error is only **.0454235**; using that approximation barely changes native residual error (.00487597 → .00487168). This does not support prioritizing the polynomial's intrinsic approximation as the explanation. It does not prove every FP32 shader arithmetic detail is irrelevant.

## Same-input down-Gemm localization

Actual native blob-87 values are now available and their instrumented out0 identities passed. For the same five fixed rows, use those **exact FP32 inputs**, exact expanded official down-projection weights [4096,12288], and a FP64 dot product. Compare against the saved native blob 88. This isolates the down Gemm arithmetic from all upstream norm, modulation, gate/up projections and GELU differences.

- **Local down-Gemm error:** L2 **.00397183908**, max **.000226008891**.
- Complete native MLP error relative to the selected-row mathematical oracle: L2 **.00487596659**.
- Upstream error propagated through the high-precision down matrix: L2 **.00279914677**.
- Cosine of local down-Gemm error with complete MLP error: **.818817125**.

This supplies evidence for a down-projection accumulation candidate. It is a local numerical opportunity, not proof of a semantic bug, a percentage attribution, or proof that correcting it improves a free-running image. The exact selected-row oracle, identities and reproducible computation are in `task-Q2-down-gemm-evidence.json` and `q2-review-scratch/audit_down_gemm.py`.

## Proposed small implementation slice and falsification

Only a **diagnostic opt-in** for the fixed graph's `gemm_6` (87 → 88), FP32 activation, K=12288, N=4096, transB=1, should change. Candidate: split-K partial sums with a compensated FP32 merge, or compensated accumulation within the down projection. Preserve weight orientation, all 12288 terms and full FP32 output; do not cast to FP16/BF16, prune terms or silently replace other Gemms. Dispatch details of the currently selected ncnn Gemm shader have not yet been traced, so this report does not falsely identify a specific shader specialization as the cause.

Predeclare the next checks before changed-math execution:

1. Same saved blob-87 inputs and fixed weights, baseline versus candidate against the current five-row FP64 oracle. Require at least a 2× reduction in both local L2 and max error on **each** input separately; otherwise reject this particular candidate. This is a candidate-screen criterion, not a replacement formal gate.
2. Add independent small adversarial dot-product fixtures (cancellation, small/large mixed magnitudes, all-negative and non-tile tails as applicable). Unsupported shapes must fail closed or take the unchanged path. Record complete output validity and resource/runtime cost.
3. If local screening succeeds, run complete block-15 official-input teacher with all output values, unchanged official reference/gates. Compare both absolute error and baseline/candidate behavior; a regression rejects promotion even if the selected rows improved.
4. Only after that evidence should root consider a single teacher step and the original full Chinese free-running fixture with its unchanged gates. Local improvement cannot close Q2. An unacceptable Vulkan resource/time cost or lack of trajectory improvement is an explicit negative outcome.

No candidate math is implemented in this report's task. This completes the requested autonomous localization to a concrete, falsifiable operator hypothesis without broad precision grids or production changes.

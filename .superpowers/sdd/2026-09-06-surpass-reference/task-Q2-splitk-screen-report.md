# Q2 diagnostic compensated down-Gemm screen

Authorized fixed candidate, no production changes: K=12288, N=4096, transB=1. Same actual blob-87 rows [0,22,4095,4096,4159] for each of official/native conditional inputs, all 4096 output columns. Baseline is the independently authenticated original native blob-88 values for exactly those rows; oracle is FP64 dot using exact native FP32 inputs and official BF16-expanded weights. This is a local candidate screen, not a formal gate.

## Implementation

`tools/diagnose_down_splitk.cpp` is standalone opt-in, linked to the frozen ncnn library. No production layer registration or ncnn source change. Weight bytes are explicitly transposed from [N,K] to raw [K,N], input [M,K], result [M,N]. 32 equal K splits of 384 terms use Neumaier accumulation plus FMA product residual. Each split stores **both sum and correction** as separate FP32 values; the second kernel performs compensated merge over both limbs. It does not truncate K or use reduced precision. Intermediate storage is O(M*N*32*2), currently M=10. The probe rejects other M outside 1..10 and all incorrect byte counts/nonfinite inputs.

v3 asserts actual uploaded buffers have scalar elempack=1 and h=1. `record_upload` selects packing using dimensions even if `use_packing_layout=false`; flattening host tensors to [M*K,1] / [K*N,1] makes raw indexing explicit. Final dimensions/type/packing are checked. Frozen .cpp/.comp/script/library/runner and every input/oracle file are in each execution/plan. The screen requires **both** L2 and max error to reduce at least 2× for **each** conditional input separately.

## Preserved attempts

- `outputs/q2-down-splitk-v1`: GLSL compilation fails on reserved loop-variable name `half`; no numerical output. Session62316, 2.1396s. Renamed to `limb` only.
- `outputs/q2-down-splitk-v2`: both real and independent synthetic tests fail. Investigation of the synthetic columns exposed automatic 4-way upload packing read as scalar rows. This is **invalid_layout**, not evidence that the compensated arithmetic is poor. All wrong outputs and logs remain. Session11410, 3.7227s. No full block was run from this result.
- `outputs/q2-down-splitk-v3`: scalar upload layout asserted, same arithmetic and fixtures. Session53911, 3.7131s, exit0; valid local screen passes. RSS sampled peak522780672B, whole GPU1748MiB, no guard stop. GPU released before follow-up implementation.

The v1/v2 files are not rewritten to hide original states. This report supplies their classification; worker `passed` is process success only, while result `passed` is the explicitly limited local screen.

## v3 complete selected-row results

| Input | Baseline L2 | Candidate L2 | Baseline max | Candidate max |
|---|---:|---:|---:|---:|
| official block14 path | .00281606516 | .000037645134 | .000226008891 | .000006834670 |
| accumulated native14 path | .00280094318 | .000036594727 | .000216686512 | .000003336099 |

L2 improvements approximately74.8× and76.5×; max improvements33.1× and65.0×. All 10×4096 elements finite. Candidate result SHA `f8e56eacfb0a53b58567557a12c832094b8b09ebffb6b74866c587bb04b41808`.

Independent synthetic fixtures cover (1e8,1,-1e8) cancellation; (2^40,1,-2^40) cancellation; alternating signs; negative sums; cross-split cancellation at positions0/383/384/12287. Four independently scaled/signed input rows include powers2^-20 and2^20. All **4×4096 output elements** exactly equal the rounded FP64 dot oracle, max0/L2=0. Synthetic result SHA `25f67ca4a68c8ee1b5bc280532ef23ed4e2685e0e98b260fda8fb231e439fd8d`. This covers fixed K/N only, not arbitrary/tail shapes; unsupported dimensions are not silently generalized.

Two CPU metric tests pass: retaining the FP64 oracle difference beyond FP32 integer precision, and including the last element of the complete5×4096 denominator. The real GPU independent fixtures are the algorithmic counterexamples; no separate unmeasured simulated shader claim is substituted for them.

v3 plan SHA `27d7433af442acb2c2517574a26530a2fd06eff7f73d1706d7cf58d0df442413`; runner SHA `57a86c48d65dafd3b81d047e43ff02766a36c8fc6066c8df744f8726e02e4549`; worker SHA `bd2f293477a86f9eb40005ec1c0e935ea7574bcdfaaf6cf1b5c2b42f7fac4607`. Package source component SHA remains `d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2`. Plans bind source tensors, weight transpose, baseline and FP64 oracle bytes independently.

## Authorized next boundary

Root has authorized complete block15 teacher only after this screen. It is still pending at this commit. Only down Gemm may change; use bounded row chunks so 4160 rows do not allocate a >4GiB split buffer. Full4160×4096 oracle/denominator and unchanged gate remain required. Row coverage, scalar layout and temporary lifetime need explicit checks. No single-step or Chinese free-running claim follows from the screen.

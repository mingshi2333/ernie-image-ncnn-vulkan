# Q2 v5 actual execution and complete-denominator independent review

Root independently reconstructed the result in a separate CPU-only process. No model forward or GPU operation was run. The reviewed v5 preparation remains d51b29d; this review covers the subsequent actual output and error decomposition reported by 929ab28/e53235a.

All 5,905 plan-bound entries were rehashed. Before/after runtime inventories each contain 2,743 files; both actual imported/mapped paths and archived bytes match. The actual result, launcher, guard, plan, source analysis, worker outcome, resource samples and input identities match the submitted evidence. New official baseline out0/75/87/88 exactly match all four old output hashes, while hook counts establish one new full official baseline and one matched MLP call; native is reused, not rerun. The old v2 swap failure remains intact.

All twelve numerical inputs were checked for full byte hashes, expected FP32 size, shape and finiteness. The five native/official comparisons use the complete 17,039,360-element denominator (51,118,080 for 87). Independent 31-row FP64 accumulation, rather than the author's 64-row implementation, reproduces every reported L2/max/NRMSE/changed count and top-row result within floating summation tolerance.

| Boundary | Complete elements | Total L2 | Input propagation L2 | Same-input implementation L2 | Cosine | Maximum reconstruction residual |
|---|---:|---:|---:|---:|---:|---:|
| 87 | 51,118,080 | 0.007343783343 | 0.010226259108 | 0.009102706389 | -0.717098036815 | 0 |
| 88 | 17,039,360 | 0.076925678345 | 0.093101540873 | 0.082305275053 | -0.621480395377 | 0 |

Each saved FP32 operand is independently promoted to FP64 before computing D=N-O, U=M-O, E=N-M. D-(U+E) is exactly zero across both complete tensors. The independent squared-norm identity residuals are -5.42e-20 and -8.67e-18, consistent with accumulation rounding. This confirms opposing terms on this fixed teacher input. It does not assign causal percentages, isolate down-Gemm rounding, prove a GELU bug, or establish a production precision policy.

All 71 actual execution samples and preflight records agree on the dedicated 10GiB memory.max, swap.max0, CPU200% and CPU0/2 affinity. Every observed PID stays in that scope with VmSwap0; sampled scope swap and memory/swap events remain0. Recomputed maxima are recursive RSS1,370,578,944B, cgroup memory.current6,919,081,984B and whole GPU3929MiB; host available never falls below18,358,583,296B. These are distinct measurements and not an exact GPU allocator or formal speed/memory comparison. The worker exits0 after37.324s, scope exit0, with no stop reason.

The independent review itself used a separate verified4GiB/swap0/CPU200% scope on CPU8/10 with CUDA hidden, completed in5.095s, observed maxRSS1,269,516KiB and unchanged zero memory/swap events. Its complete recomputation script and JSON are stored beside this report. The next 84-only experiment may be prepared under the separately stated contract; it has not been executed or declared necessary for a production change. Full-trajectory quality and formal acceptance remain open.

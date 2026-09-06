# Compensated QK experiment: do not promote

This real-input diagnostic evaluates extending the existing Kahan attention shader to the Q·K score reduction, with native transpose, scale, mask and GQA specializations. It applies only to FP32, non-cache attention; the existing low-storage and autoregressive cache paths are unchanged. The candidate is archived as a patch and frozen source identity, then removed from the production source tree because it does not meet the predeclared progression criterion.

## Actual execution

Both binaries run the exact historical Chinese step-0 block-15 teacher fixture, ten input files and unchanged full model weights. The OFF output reproduces historical SHA `175bfa394aec3b7e576caaea3fd897cf84db122d4cd9378d7d72dfe704df68a3` bitwise. The expected result is the saved official block-15 output `a3f77e846934a6fe7085bf366dd21ef78aa779b702c5c97ba237834a0635a197`.

| Complete 17,039,360-value comparison | OFF baseline | ON candidate |
| --- | ---: | ---: |
| NRMSE | 6.011371945074345e-7 | 5.865782620340838e-7 |
| Error L2 | 0.1987765447421232 | 0.19396237866052207 |
| Maximum absolute error | 0.006591796875 | 0.0068359375 |
| Finite output values | 17,039,360 | 17,039,360 |

Overall L2 decreases about 2.42%, but maximum error increases about 3.70%. The plan required improvement in both before attempting a full image. No full-image run or production promotion follows this result. It does not locate the root cause of complete Chinese or fixed-1376 trajectory failures.

Both variants also pass the same 23 actual operator cases: one long-attention FP64-oracle case, four bounded/unbounded mask/GQA cases and eighteen FP32/FP16/BF16 cache lifecycle cases. All command exit statuses are zero, with no skips. Two initial review parsing mistakes counted the cache metadata header as a case; explicit metadata and 1/4/18 case enumeration fix the review without rerunning any model.

The 353 bound entries are verified before and after each phase, including 311 project source files, the saved official input/oracle, weight files, frozen executables and linked ncnn archives. This is a source/binary/input identity record; it is not a sealed dynamic-driver closure or a formal performance test.

## Resources and memory interpretation

Both phases use a dedicated 16 GiB cgroup, swap disabled, CPUs 4/6 with a two-core CPU quota, a continuously sampled 3 GiB host-availability floor and 6144 MiB whole-device GPU ceiling. Each completes in about 8.9 seconds, including operator probes and identity verification. OFF/ON sampled cgroup peaks are 642,048,000 / 632,696,832 bytes, host-available minima 17,160,519,680 / 17,487,196,160 bytes, and whole-device GPU peaks 4519 / 4096 MiB. Both memory.max and OOM event counters remain zero. These timings and sampled peaks are diagnostic only.

This experiment has no allocation failure. Changing arithmetic changes the result on identical saved inputs while the OFF run reproduces the old result exactly; it supplies further evidence for arithmetic differences rather than memory-capacity failure in this comparison. Earlier actual allocation/OOM failures remain separate evidence. It does not establish that every historical failure had the same cause, nor diagnose physical RAM health.

Raw tensors, model weights, binaries and sampled process streams remain under `outputs/q2-compensated-qk-v1/`; this artifact stores only small reviewable records. Existing full-image quality failures and thresholds remain unchanged.

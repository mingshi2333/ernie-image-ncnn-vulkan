# Actual allocation CLI: one 64×64 ON/OFF pair

The native 64×64 apple development fixture completed once per build, with identical FP32 text/DiT/VAE/scheduler, 8 steps, CFG1, native 15-token text (bucket32, default Gemm), CPU2, GPU0 DiT, CPU direct VAE, saved CHW FP32 noise, PE off and tracing enabled. No saved embeddings bypass was used.

All 25 FP32 trace tensors, IDs, exact prompt (27 files total), and PNG are bitwise identical. PNG SHA256: `44ea37d9f2f73dc64a4c510d9fc262c7b02dffb689f67010bb34c676025f81ba`. Complete combined stdout/stderr logs match after removing only the explicit ON diagnostic banner, durations on known progress lines, and side-specific image output paths. Streams were combined at capture; separate stdout equality is not claimed.

ON observed 2375 actual VkDeviceMemory allocations, a simultaneous peak of **968724992 bytes**, and zero live bytes after actual cleanup. All 1254 allocator generations are inactive; initial/final Vulkan instance absent; device identity present; valid and coverage_complete true. Coverage is observed ncnn allocator lifetime, not all driver/process/device memory. Per-class peaks must not be summed.

Parent guarded elapsed times were ON368.486s/OFF368.159s. These are trace-on diagnostic times, not a speed result. Parent RSS-sum peaks were ON1119825920/OFF1220976640 bytes, whole-card sampled peaks2688/2845MiB, minimum host available17842704384/14867070976 bytes. Neither run triggered the18GiB RSS-sum/6144MiB whole-card/1GiB host-available guards. Whole-card sampling is separate from actual allocation statistics.

The earlier new-hook bounded real allocator probe also passed: ON/OFF1024 bytes bitwise equal, peak2134016 bytes/11 allocations, new device identity present and live0 after instance destruction. Raw logs/events and complete allocator records remain under `outputs/allocation-real-o1-v3` and `outputs/allocation-pipeline64-o1-v1`, bound by the included manifests and hashes.

This establishes instrumentation output preservation and allocator lifetime coverage for this one development fixture. It does not close full O1 stage metrics, broad quality, formal performance/memory gates, or S. CPU RSS/GPU time/stage timing fields in the CLI report remain explicitly unavailable.

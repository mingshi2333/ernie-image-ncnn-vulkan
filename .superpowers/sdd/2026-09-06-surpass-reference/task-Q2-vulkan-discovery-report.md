# Q2 actual no-model Vulkan discovery and up84 v4 CPU freeze

Preserved22dd472 and all three failed model attempts. A new minimal source calls only create_gpu_instance, get_default_gpu_index/get_gpu_device and destroy_gpu_instance, with two-second observation holds. No Net, tensors, load_param/load_model, shader dispatch or forward. Same libncnn.a and original link closure; v1 compilation put source after static archive and failed unresolved symbols, preserved. v2 compilation moved source before archive and succeeded. Compilation and execution logs are separate.

Execution session17032 exit0, default device0 NVIDIA RTX4060 Laptop, lifecycle BEFORE_CREATE→DEVICE_READY0→DESTROYED. Dedicated4GiB/swap0/quota200%/CPU0,2 with host3GiB/GPU6144MiB/30s supervisor; all observed unknown paths were discovery evidence, not model acceptance. GPU released immediately after completion. Full93 mapped paths were hashed at first observation and again after exit; all identical. 86 samples, actual resources below.

{
  "status": "discovery_completed",
  "result_sha256": "efdc06e9afb9e0942007c40987931d5ff8d2cfd115020994247a21d1d0f75d4f",
  "runner_sha256": "313eec2fd57796f65ba4d448f4bb5caa0c6b94fa5b8dacae8e35031a5f9e430c",
  "source_sha256": "2f3e6b6990efde5f29d02767fea3766a81c4c4adc1c33b640a9f0849ed5b3ba7",
  "wall": 7.68990574000054,
  "mapped_files": 93,
  "samples": 86,
  "peak_rss": 226660352,
  "peak_gpu": 1765,
  "host_min": 17687711744,
  "swap_max": 0,
  "all_mapped_post_sha_equal": true,
  "model_forwards": 0,
  "environment_limit": "Vulkan/XDG/LD/GLVND selectors inherited identically; model-child CUDA/OMP/BLAS/MKL overrides were not copied, so not an exact complete environment replay."
}

Important scope qualification: probe inherits the same Vulkan/XDG/LD/GLVND selection environment, but did not explicitly copy the model guard child overrides CUDA_VISIBLE_DEVICES=0/OMP_NUM_THREADS=2/OPENBLAS_NUM_THREADS=2/MKL_NUM_THREADS=2. CPU affinity/quota were exact. No numerical work occurred, but do not claim all environment bytes were identical. Root was notified; no unapproved second lifecycle run was started.

Previously unbound EGL dispatch/Mesa/NVIDIA libraries were actually observed. Installed GLVND vendor50_mesa.json and10_nvidia.json directly name the Mesa/NVIDIA sonames; their current exact files and dependency closures are now frozen alongside EGL external-platform JSON directories and __EGL/__GLX environment identity. Maps alone do not prove which caller dlopened each library. Other newly observed paths include locale data/gconv cache and the probe executable; all are retained. Actual future lazy loads remain subject to unchanged unknown-mapping rejection.

New candidate model preparation outputs/q2-up84-v4 binds6312 identities; all6251 prior entries remain. Native/official/sequence/guard/resources are unchanged. Five loader and three up84 tests pass; fullbound+environment/config verification passed in4GiB/swap0 CPU scope. No modelattempt in v4 has run. The discovery environment qualification must be considered in root review before scheduling.

- plan.json: `a06a043311cd85ebdcf1c551fbcc9adadc4253140c538b17130d0155bc88ea89`
- launcher.py: `63fe37905daf4cad497eb5a3033c941c810f24508702f73f67c274e7b0e1d09d`
- execution/runner: `ec5de0277fb94beab136877253b44873aa44e0285a145c50598cc51981c5f720`
- execution/diagnose_up84_scope_guard.py: `86b50e9d9024ea2f533d5b2ccb1a437bdfb43ab6594ec7bbf01f52e3d07b0193`
- execution/diagnose_up84_official.py: `ba08f5af9914f475bb93b9a011c03be4707ce6e154c4bd67d447c6bd79d0f630`
- execution/diagnose_up84_sequence.py: `6c44698d50cb9cab988cb877605683c9660d8b3125b533983add5b5972ffc578`
- execution/diagnose_vulkan_inventory.py: `593e4b0e01b203847ddb492c2e23d8c9b07e82cdfa2e2e43d7979d840ac2bc5d`
- execution/loader-inventory.json: `bd639a19dc37bbc013a9e14f8f471b274bb0dbddf4416d0d069269eca0c890b1`
- bound_count: `6312`

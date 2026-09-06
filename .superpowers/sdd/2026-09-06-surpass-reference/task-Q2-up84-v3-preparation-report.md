# Q2 up84 v3 — loader cache/configuration closure, CPU only

Preserves failed v2 and87289db. No model retry or Vulkan call. New attempt outputs/q2-up84-v3. Only preparation identity changes: include actual /etc/ld.so.cache and ld.so.conf recursive include inputs, optional ld.so.preload, and glob membership (including currently absent files) in a separately labelled `dynamic_loader_cache_and_configuration_not_ELF` section. These24 extra inputs are not misclassified as ELF libraries. The guard still authenticates the original `.so` mapped-path set without any exclusion, and now its existing inventory preflight additionally verifies cache/configuration identity. No resource, model, math, endpoint or maximum-two-forward change.

Native runner, sequence, official payload and guard are exactly the same bytes as v2. New directory copies and the collector/probe account for the additional bound records. The old failures remain failures; post-stop cache observation does not retroactively authenticate them. Unresolved optional shader-debugger remains explicitly unresolved; no system layer settings changed.

## Actual CPU Python process probe

Executed a real new Python subprocess under dedicated scope10GiB/swap0/CPU200%/affinity0,2. It imports no model or Vulkan code and explicitly mmaps the loader cache for one second so that the original mapped-path observer can test it deterministically (ordinary startup may unmap the cache between samples). The same guard sampling/limit functions and mapped-path/hash checks authenticated the observed cache and libraries. This is an actual-process observer check, not a Vulkan/model readiness guarantee. Process exited0 and scope ended normally; zero forwards and zero Vulkan calls.

{
  "status": "CPU_python_startup_and_cache_mapping_pass",
  "child_pid": 2212471,
  "samples": 47,
  "observed_file_count": 29,
  "cache_observed_sha256": "46665a1af23253ad75c3c57f29ea75f9a4608f4b072e8d1ed7cbb5b9038b8220",
  "peak_rss": 38637568,
  "swap_max": 0,
  "host_min": 18180571136,
  "model_forwards": 0,
  "vulkan_calls": 0
}

## Frozen approval identities

- plan.json: `b69caa2b2e1d33025e2c76088d80876bb2a331e284de9b7367cab7099765c82a`
- launcher.py: `cf6aad76cdb199c24cec4ca831d91f748e875eb3d769fb892ce9427a57b125f6`
- execution/runner: `ec5de0277fb94beab136877253b44873aa44e0285a145c50598cc51981c5f720`
- execution/diagnose_up84_scope_guard.py: `86b50e9d9024ea2f533d5b2ccb1a437bdfb43ab6594ec7bbf01f52e3d07b0193`
- execution/diagnose_up84_official.py: `ba08f5af9914f475bb93b9a011c03be4707ce6e154c4bd67d447c6bd79d0f630`
- execution/diagnose_up84_sequence.py: `6c44698d50cb9cab988cb877605683c9660d8b3125b533983add5b5972ffc578`
- execution/diagnose_vulkan_inventory.py: `18c6ace40ab57c670a2b7f5b8bdf648a3f8a4967cec257936b72b49e65feb09b`
- execution/loader-inventory.json: `6166209ab1e3985fb6ff4dfb9b4c8f799455113817dc17150e01388c530f26b5`
- execution/diagnose_loader_startup_probe.py: `7953d800b4a1da0c5d81bdda5c1d3eb7742d295bb38a5f6342e1ccbc92ab7562`
- bound_count: `6251`

All6251bound plus environment/directory/configuration identities were rechecked CPU0,2 under4GiB/swap0 scope. Five loader tests (including cache mutation and include membership change) plus three existing up84 invariant tests pass. Actual evidence includes python-startup-probe.json and startup-summary.json. Root independent review and a new explicit model slot are required; this preparation has no84 numerical outputs.

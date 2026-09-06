# Q2 up84 v2 — CPU-only replacement preparation

Preserves v1 as execution_identity_failed. No model/GPU rerun occurred. New output: outputs/q2-up84-v2. Only the loader inventory/preflight and output identity change; native runner, official payload, sequence and all tensor/weight/math inputs are unchanged. Maximum native fullblock1 + officialMLP1; original full-output prerequisites and resource guards unchanged.

## Discovery correction and limits

New tools/diagnose_vulkan_inventory.py scans every JSON under Linux configuration/data Vulkan roots, explicit/implicit layer roots and environment-specified paths (as a conservative superset). It includes /etc, XDG configuration/data and /usr/local/share; no `.x86_64.json` filename filter remains. ELF header class decides32-bit exclusion. Absolute library paths remain absolute, paths containing slashes are resolved relative to the manifest, basenames use frozen LD_LIBRARY_PATH/cache candidates, ambiguous64-bit candidates are recorded as unresolved. Each resolved library has its ldd closure recorded and hashes bound. Both /etc AMD manifests now bind the actual amdvlk64.so library and closure.

Search rules were checked against pinned Khronos Vulkan-Loader v1.4.341 LoaderDriverInterface.md and LoaderLayerInterface.md, plus LoaderSettingsFile.md and loader.c (copies in execution/loader-docs). The installed Fedora package is vulkan-loader1.4.341.0-1.fc44; actual loader string inspection exposed /etc and both implicit-layer override variable names. Upstream tag source is primary rule evidence, not proof that Fedora binary has no downstream changes. Search inventory is explicitly a superset, not exact loader-order/filter emulation or a guarantee of all transient mappings.

Directory membership, loader environment (including activation variables named in manifests), settings locations, manifest bytes and resolved closure bytes are recorded. The guard checks these again before any model process. The new guard only adds this preflight; its unknown actual-mapped-library rejection is byte-for-byte the same logic and thresholds. It does not alter system settings or driver selection.

Unresolved: installed NVIDIA shader-debugger relative path ../../../libShaderDebuggerAPIInjection.so. It remains explicit and does not authorize any unknown mapping. No layer-enable bypass/exclusion was retained. No current loader_settings.d file was discovered; future membership changes fail. Catalog variants1/2/3 in the separate post-audit directory preserve the preparation history; variant2 explored activation-based exclusion but is NOT the frozen execution catalog. The final catalog follows the existing reviewed contract: unresolved optional layers remain listed, actual unknown mappings failclosed.

## Frozen entry and tests

- plan.json: `9b04ffbe1c580e5227120865c0b2b8ad9defe15dd9f8ae7596e29356ab532562`
- launcher.py: `392e10a9f813eacb6597bddcc98b3fc784f38500a1459c18a60186c4bcd2db31`
- execution/runner: `ec5de0277fb94beab136877253b44873aa44e0285a145c50598cc51981c5f720`
- execution/diagnose_up84_scope_guard.py: `86b50e9d9024ea2f533d5b2ccb1a437bdfb43ab6594ec7bbf01f52e3d07b0193`
- execution/diagnose_up84_official.py: `ba08f5af9914f475bb93b9a011c03be4707ce6e154c4bd67d447c6bd79d0f630`
- execution/diagnose_up84_sequence.py: `6c44698d50cb9cab988cb877605683c9660d8b3125b533983add5b5972ffc578`
- execution/diagnose_vulkan_inventory.py: `44621babbad602ba3910295a3f1e60e66846f7bbf822d23c7bbc42d8571658dd`
- execution/loader-inventory.json: `5c5c9ad0e8877d684b00f6d5ad93d42a023ca36e11b7d907afe1ffcbebc35847`
- bound_count: `6198`

All6198 bound identities and loader environment/directory inventory were checked in a CPU0,2, 4GiB/swap0/200% scope. Four collector tests passed (/etc/non-architecture filename; absolute/relative/basename and ELF32; implicit overrides/settings roots; unresolved/membership failure), plus three existing up84 invariance tests. No new84 values exist. New launcher requires root independent review and a new explicit model/GPU slot before execution.

Primary references: https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.4.341/docs/LoaderDriverInterface.md ; https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.4.341/docs/LoaderLayerInterface.md ; https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.4.341/docs/LoaderSettingsFile.md .

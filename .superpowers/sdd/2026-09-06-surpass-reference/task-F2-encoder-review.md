# F2 native VAE encoder boundary independent review

## Scope

Reviewed commit `6ed080b`, the actual 32x32 and 64x32 output directories, and `.superpowers/sdd/2026-09-06-surpass-reference/task-F2-encoder-evidence.json`. Only small JSON, parameter graph, input, expected-boundary, and actual-boundary files were hashed or recomputed. No large weight scan, export, model run, or GPU process was performed.

## Verdict

No Critical finding. One Important validation-integrity gap must be closed before treating this helper as a production package/migration verifier. The existing frozen runs themselves are internally consistent and support the narrower claim that the two exported static encoder graphs match their official-module boundary fixtures within the declared gates.

## Important

### I1. The verifier does not authenticate the semantic/source identity it records

`tools/validate_img2img_encoder.py:24-35` checks the file inventory and self-consistent file hashes, ncnn/official revisions, and equality of the two embedded weight dictionaries. It does not enforce `schema_version`, `manifest.source_sha256`, `fixture.posterior`, `fixture.packing`, the RGB normalization label, `diffusers_revision`, `vae_config_sha256`, `official_source_sha256`, `distribution_source_sha256`, or `source_manifests`. Later validation checks the boundary names and BN metadata, but still does not bind most of these source/semantic identities (`tools/validate_img2img_encoder.py:119-138`).

A small synthetic reproduction copied only the small fixture files, replaced `head.ncnn.bin` with an empty file, set `source_sha256` to 64 zeroes, changed `posterior` to `sample_unreviewed` and `packing` to `unreviewed`, and recomputed the manifest's self-consistent hashes. `verify()` returned successfully:

```text
ACCEPTED_TAMPERED_PROVENANCE sample_unreviewed unreviewed 0000000000000000000000000000000000000000000000000000000000000000
```

The empty graph binary is used only to keep this review reproduction small; the same acceptance follows for a self-consistent replacement binary because no trusted graph/source identity is consulted. A production migration path must bind an approved manifest or graph/source hash and validate every semantic field that controls mode/packing/normalization/BN. Tests should mutate each independently and require rejection.

This does not invalidate the current frozen evidence: all five changed source files, the installed pinned Diffusers source files, the peer encoder parameter file, both small native parameter graphs, and both model manifests matched the hashes in `task-F2-encoder-evidence.json`. The defect is that the reusable verifier cannot prove those identities for a future package by itself.

## Verified evidence

- The reference wrapper implements `encoder -> quant_conv -> first 32 channels (posterior mean/mode) -> pixel_unshuffle(2) -> BatchNorm`, and compares all three outputs bitwise against `model.encode(x).latent_dist.mode()` plus explicit packing/BN (`tools/export_vae_encoder.py:64-86`). Installed pinned Diffusers constructs the posterior by splitting channels into mean/logvar and `mode()` returns mean. There is no posterior sampling.
- RGB normalization is explicitly FP32 `(uint8 - 127.5) * (1/127.5)`, then HWC-to-NCHW (`tools/export_vae_encoder.py:31-34`). The unit test checks the operation order at the bit level.
- The encoder config and live module require BN epsilon `1e-4`, affine false (`tools/export_vae_encoder.py:55-74`). The fixture separately records decoder inverse-BN epsilon `1e-5`; current native decoder unpack uses `sqrt(variance + 1e-5)`. The asymmetry is preserved rather than normalized away.
- The actual ncnn graphs contain `Crop -> Reorg(pixel_unshuffle 2) -> BatchNorm eps=0.0001`. The 32x32 and 64x32 graphs differ in the two attention reshape dimensions, proving these are separately specialized static graphs rather than one demonstrated runtime-dynamic graph.
- Small-file hashes for `fixture.json`, `input.rgb`, `in0.f32`, `out0/1/2.f32`, `head.ncnn.param`, and `model.json` match their manifests and the tracked evidence for both shapes. The separately stored 32x32 actual outputs also match all three tracked actual SHA256 values.
- Actual native error evidence is strong for its narrow scope:

```text
shape   boundary    NRMSE                 max abs
32x32   mean        6.0948901118e-7       3.3378601074e-6
32x32   packed      6.0948901118e-7       3.3378601074e-6
32x32   normalized  6.1170072912e-7       1.9073486328e-6
64x32   mean        6.9927392897e-7       4.7683715820e-6
64x32   packed      6.9927392897e-7       4.7683715820e-6
64x32   normalized  7.0029977433e-7       2.7418136597e-6
```

The recorded reference/convert/native processes were cgroup-observed under the 2 GiB maximum and used two-thread limits. The 32x32 top-level export-only result honestly says validation pending; its native validation is retained separately and explicitly referenced in the tracked evidence. The 64x32 top-level result includes its passing validation.

## Test evidence

```text
python3 -m unittest tests.test_img2img_encoder
Ran 6 tests in 0.008s - OK
```

These tests cover small dimension bounds, exact deterministic RGB identity and FP32 normalization order, invalid image tensors, a passing and failing numerical gate, prelaunch low-memory refusal, and no unsupervised fallback when `systemd-run` is unavailable.

## Production migration boundary

The two static exports establish that the complete encoder and the mean/packing/BN tail are convertible and numerically sound at 32x32 and 64x32. They do not establish a graph for 512x512 or other production dimensions, a reviewed runtime reshape allowlist, shared schema-3 packaging, or end-to-end encode/noise/denoise/decode quality. The two parameter files visibly contain shape-specific attention reshapes, so production integration must use a separately reviewed specialized graph or F1's authenticated shape mechanism. This evidence cannot by itself justify reusing either tiny graph at arbitrary production sizes.

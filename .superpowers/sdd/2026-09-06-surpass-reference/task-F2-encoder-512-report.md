# F2 512x384 encoder extension: preparation report

## Completed static review

The independently validated 32x32 and 64x32 encoder params differ only at two spatial reshapes. For 512x384 input, the reviewed substitutions are:

- `reshape_77`: flattened latent spatial count `16 -> 3072` (`64 * 48`), while channel width remains 512.
- `reshape_78`: spatial shape `4,4 -> 64,48`, while channel width remains 512.

No convolution, attention, GroupNorm, Crop, Reorg, weight, input/output name, or other topology field is changed. `specialize_vae_encoder.py` requires the already validated 32x32 template hashes and exactly these two original reshape records; it rejects any other large shape or unexpected reshape.

`reference_vae_encoder_large.py` is a separate official fixture producer fixed to 512x384. It loads the pinned official encoder/quant/BN modules, uses posterior mode, pixel-unshuffle 2, encoder BN epsilon 1e-4, two CPU threads, and saves source RGB plus official mean/packed/normalized FP32 boundaries. It does not trace, invoke pnnx, denoise, or claim acceptance. The original `export_vae_encoder.py` remains byte-for-byte unchanged so its recorded source identity and the existing reviewed 32x32/64x32 evidence remain valid.

## Tests

- `python3 -m unittest tests.test_vae_encoder_specialize`: 3/3 passed.
- `.venv/bin/python -m unittest tests.test_img2img_encoder`: 12/12 passed.
- Both existing reviewed candidates were reverified after this change.
- Python compilation and diff whitespace checks passed.

## Resource gate and actual execution

The first resource check showed about 19.97 GB MemAvailable, but root then started the authorized full native PE/vector-text 512x384 run. During that run its native process reached about 13.9 GiB RSS and host MemAvailable fell to about 5.84 GB. Therefore the separate encoder job was not started. This follows the requirement to avoid overlapping the 3 GiB encoder scope with the large PE run and preserves the pending state honestly.

After the PE memory peak passed and root explicitly released the CPU queue, the official fixture ran successfully in a 3 GiB `MemoryMax`, no-swap user scope with two CPU threads. It produced shapes `[1,32,48,64]`, `[1,128,24,32]`, and `[1,128,24,32]`; every tensor was finite and matched its recorded size/hash. Fixture SHA-256: `ee5ce5db3cb9912ff5e824bf062d1376bcf9b2e91e942d06924ecd187a93e1a9`.

Specialization changed exactly the reviewed two records. The resulting param SHA-256 is `4858cc8b3f2ed267316c122ecef538ae4217c778a48e8977768618833ac5af92`; weights remain `7fa2441a94886d9a1d44dbafe4fbac9211190e342b1cac171acb94c0faf517ce`.

The rebuilt native runner (SHA-256 `c587af1636d592bd2dc00d417a5c958b02738a76942a8e4391ee6210dbf9bbe0`) then ran the complete 512x384 strength-zero CPU/direct chain under the same 3 GiB/no-swap limit. The existing portable decoder param/bin/BN files were independently checked against their package manifest before use.

| boundary | max abs | mean abs | NRMSE | cosine |
| --- | ---: | ---: | ---: | ---: |
| mean | 6.55651e-6 | 8.08893e-7 | 7.02136e-7 | 0.999999999999755 |
| packed | 6.55651e-6 | 8.08893e-7 | 7.02136e-7 | 0.999999999999754 |
| normalized | 3.69549e-6 | 4.57937e-7 | 7.06593e-7 | 0.999999999999751 |
| start | 3.69549e-6 | 4.57937e-7 | 7.06593e-7 | 0.999999999999751 |
| unpacked | 6.55651e-6 | 8.09763e-7 | 7.04565e-7 | 0.999999999999753 |
| decoded | 1.65403e-5 | 8.19940e-7 | 2.13508e-6 | 0.999999999997726 |

All tensor boundaries pass the existing FP32 gates. The final native PNG differs from the official PNG by at most one 8-bit value, with mean absolute pixel error `0.000123766`; native/official PNG hashes are respectively `6bce341e6453a8efb84f6e5fbd87b545d4d5e29f6b3666840bf4cc58f2ed9931` and `984d9a030f47368f71d38e145594dfb75edf0744a14a914878f14d522e65c1d9`.

Evidence inventory manifests:

- official encoder fixture: `bd0fd9f20057d012c6fbb7725981cc7ab99a1cdd193410a564944a750e9e0c50`
- specialized encoder: `ad6206ccd140351c2543afa123837da32ef84ee7ccaf43988071ef218c7d7099`
- official reconstruction: `e1806c3b0ca42a57fca2a365ab4fc65af4ba724258b65adea245c8fa49d9ded3`
- native reconstruction: `b1eab78938bfa7db4547b09382dab89865e52e0b68d941f85b7d52bd48812d92`

No formal15 input or GPU encoder work was performed. This result validates one deterministic 512x384 development fixture and does not establish 15-case acceptance or production schema3 encoder inventory.

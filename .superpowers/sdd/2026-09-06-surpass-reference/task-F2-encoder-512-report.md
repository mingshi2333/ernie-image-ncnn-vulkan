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

## Resource gate and pending execution

The first resource check showed about 19.97 GB MemAvailable, but root then started the authorized full native PE/vector-text 512x384 run. During that run its native process reached about 13.9 GiB RSS and host MemAvailable fell to about 5.84 GB. Therefore the separate encoder job was not started. This follows the requirement to avoid overlapping the 3 GiB encoder scope with the large PE run and preserves the pending state honestly.

The official 512x384 fixture, specialized encoder graph hash, native three-boundary comparison, strength-zero reconstruction, and decoder comparison remain pending until root releases the memory queue. No formal15 input or GPU encoder work was performed, and this preparation is not a 15-case or production schema3 claim.

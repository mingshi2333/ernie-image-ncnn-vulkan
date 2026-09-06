# Q3 bounded PE contract report

This delivery is limited to the real native sampler contract and frozen development-batch input identity. It did not load or run the official or native full PE model, did not use a GPU, and did not run the formal 72-case set.

## Contract completed

- `tests/test_pe_sampling.cpp` calls the production `ernie::sample_pe_token` entry point. With logits `[log(.7), log(.2), log(.1)]`, temperature 1, top-p 1 and local seed 20260906, it checks 100000 draws and requires every observed frequency to be within 0.01 of its declared probability.
- The same production entry point proves top-p 0.6 retains only the 0.7 token, preserves the existing first-index greedy tie behavior, preserves fixed-seed reproducibility, and rejects invalid temperature, top-p, max-token, empty-logit and nonfinite-logit inputs.
- `tests/fixtures/pe-development-batch.json` freezes 12 development inputs with the required prompt, WH dimensions, max tokens, mode, temperature, top-p, actual formatted input token count and input-ID SHA256. It includes English, Chinese, Japanese, quotes/backslash, mixed newline forms, leading/trailing whitespace, empty/control inputs, an output limit of 2048, and an actual 2048-token near-capacity input.
- `tools/validate_pe_tokenizer.py` builds and validates the batch using the pinned local official tokenizer assets. It checks the manifest checksum, exact field set, unique IDs, dimensions, limits, explicit greedy settings, token capacity and recomputed input-ID identity. An over-capacity input is rejected during tokenizer/manifest validation without invoking a model.
- `tools/reference_pe.py` can select one frozen case with `--batch-contract ... --case-id ...`; it validates the complete batch before any model construction and binds the saved official reference to the suite, case and input-ID hash.
- `tools/validate_pe.py` optionally requires a reference to match a validated frozen case before starting the native runner. Existing greedy logits gates remain `nrmse/atol/rtol = 2e-4`; a completed native result records the batch case identity.

## Verification

```text
cmake --build build-dev --target ernie-pe-sampling-contract -j2
[100%] Built target ernie-pe-sampling-contract

ctest --test-dir build-dev -R pe_sampling_contract --output-on-failure
1/1 Test #9: pe_sampling_contract ... Passed
100% tests passed, 0 tests failed out of 1

.venv/bin/python -m unittest tests.test_pe_batch_contract -v
Ran 4 tests in 0.963s
OK

.venv/bin/python -m py_compile tools/reference_pe.py tools/validate_pe.py tools/validate_pe_tokenizer.py tests/test_pe_batch_contract.py
exit 0
```

The committed frozen batch validates against the actual pinned tokenizer. Its manifest SHA256 is `6b97fecef68035c82d105ff08a54f88e9f6862b9e23f0a2fdb452669302dc52a`; the near-capacity case has exactly 2048 formatted input tokens.

## Pending real acceptance

All 12 cases explicitly remain `pending_real_official_and_native_pe`. Complete official greedy outputs, native logits/token/text comparisons, EOS behavior, output-length stopping, cache history/session interleaving and full-model precision evidence require later CPU model runs. No result in this change claims those checks passed. The formal 72 inputs remain unrevealed and unexecuted by Q3.

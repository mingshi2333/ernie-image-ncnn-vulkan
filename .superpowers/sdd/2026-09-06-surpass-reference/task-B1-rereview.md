# B1 fix independent rereview

Review scope: fix commit `c8835ee`, limited to the two Important findings in `task-B1-review.md` and the documented resolution of its Minor finding. No model or GPU process was run. The existing frozen corpus was verified in place and was not modified.

## Critical

None.

## Important

None. Both previously reported Important findings are closed.

### Closed: ambiguous prompt/noise sources

`freeze_inputs` now requires exactly one of `prompt`/`prompt_file` and exactly one of `seed`/`noise_file` before the staging directory is created (`tools/acceptance_manifest.py:98-111`). The new field-by-field behavior test covers both ambiguous combinations and also proves that rejection leaves no destination directory (`tests/test_acceptance_manifest.py:72-81`). This closes the provenance ambiguity rather than merely documenting precedence.

### Closed: incomplete enabled-PE semantics

The common `_validate_case` path now requires `stop_at_eos`, `add_generation_prompt`, and `template_source` together with the previously required PE identity fields (`tools/acceptance_manifest.py:46-65`). It validates true integer token limits in `[1,2048]`, boolean EOS/template flags, a nonempty template source, a lowercase SHA256, and explicit finite greedy `temperature=0`/`top_p=1`. Because `freeze_inputs` and `verify_inputs` share this validator, the checks apply both at creation and when reading an externally supplied manifest. The tests remove every required field, substitute invalid types/values, and prove that recomputing the outer manifest checksum does not bypass semantic validation (`tests/test_acceptance_manifest.py:82-105`).

## Minor

No remaining blocking issue. The earlier unlisted-file observation is now resolved as an explicit contract rather than a closed-directory policy: module and verifier documentation state that only manifest-listed inputs are authenticated and consumers must resolve inputs exclusively from those entries (`tools/acceptance_manifest.py:2-9,150-155`). The test demonstrates that a later report is outside the authenticated input set (`tests/test_acceptance_manifest.py:106-112`). This is consistent with the original review's allowed remediation and supports adding diagnostic reports after freezing without presenting them as inputs.

## Verification evidence

```text
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -p 'test_acceptance_manifest.py' -v
Ran 19 tests in 0.068s
OK

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python tools/acceptance_manifest.py --verify outputs/port-corpus-v1
All frozen input identities verified
```

## Assessment

Commit `c8835ee` closes both Important API findings without rewriting the anchored B1 manifest or protocol. The actual 101-case frozen corpus still passes complete offline identity verification. This rereview does not claim any model, image-quality, or performance acceptance.

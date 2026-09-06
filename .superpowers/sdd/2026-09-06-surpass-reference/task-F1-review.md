# F1 independent review: 77049c7 and 0610368

## Verdict

**Pass for the implemented offline/static scope.** I found no Critical or Important defect in the two reviewed commits. The implementation remains deliberately incomplete for the actual F1 runtime goal and its reports state that boundary accurately.

This review used source inspection, small unit tests, and existing JSON/hash evidence. It did not copy full weights, instantiate a model, run a graph, use a GPU, or rebuild while another task owned `build-dev`.

## Critical

None found.

## Important

None found.

## Shape planner

`src/shape_plan.cpp:19-44` validates signed inputs before conversion, restricts each axis to 16..2048 in multiples of 16, caps image area at 2,097,152 pixels, limits valid text to 1..2048, and uses checked `size_t` multiplication/addition for every exposed count. It distinguishes valid text tokens from the selected 32/64/2048 bucket. The output is count metadata only: it does not create a graph, RoPE, mask, tensor, or runnable request.

The planner can describe totals outside the current runtime's 6144-token limit. This is not an acceptance bypass because it is an independent library and is not linked into the generator; the report explicitly identifies that distinction and does not claim those shapes execute.

## Static graph contract and model configuration

`src/shape_graph.cpp:32-95` accepts source and target configurations only after both pass the existing numeric bounds and the three-value `reviewed_shape_config` allowlist. The three allowed configurations correspond to the pinned 64x64, 512x384, and 1024x1024 static packages. Unknown shapes therefore fail before graph rewriting.

Every graph kind has an exact node/parameter allowlist. The implementation requires each allowed field exactly once with the expected source value, rejects duplicate node names and malformed arity, substitutes formula labels, and verifies the SHA256 of the complete normalized graph against a fixed kind-specific hash. Unknown nodes, operators, parameters, topology, or omitted required fields change the complete hash or fail directly. Text graphs cannot cross text buckets; an independently exported template remains required.

`validate_model_config` preserves the earlier axis, text relation, and total-token bounds. The existing `model_config` parser still requires exactly the same six keys, text_layers=25 and dit_layers=36. The reviewed commits do not broaden runtime model acceptance.

The stored static audit contains three packages and 192 graph records: one pinned schema-1 package and two pinned portable schema-2 packages, 64 graphs each. Each live manifest byte hash still matches the hash recorded by the audit. Every stored record says `runtime_generation_validated=false`, and the overall artifact says `dynamic_instantiation_supported=false`.

The stored native graph crosscheck has 48 records: 39 generated texts exactly match an existing target graph, six cross-text-bucket attempts are rejected, and three corruption/unknown-shape attempts are rejected. Its scope is explicitly `C++ graph text only; no weights or graph execution`.

## Tokenizer SHA ABI

The new `ernie_shape_sha256` ABI is isolated to hashing graph bytes. It rejects a null output, rejects null input for nonzero length, caps input at 1 MiB, catches Rust panics, and writes exactly 32 digest bytes on success. The empty and `abc` known vectors are recorded as passing. It uses the already pinned `sha2` dependency and does not call or alter tokenizer/package validation.

The recorded source-hash JSON matches every reviewed file exactly at commit `0610368`, including unchanged `tokenizer/src/package.rs`. Native package schema handling therefore remains in the old package validator rather than the new hash ABI.

## Offline shared-object candidate

`tools/package_dynamic_model.py` intentionally writes `contract.json`, never `manifest.json`. The policy is exact and type-checked: candidate schema 3, runtime unsupported, quality pending, pinned static instances only. Adding a runtime manifest is rejected. Existing native validation separately records `Unsupported package schema` for a small fake schema-3 package.

The builder accepts only one or two pinned, portable schema-2 source manifests. Schema-1 migration is explicitly rejected as pending. Before copying, it runs the static graph audit and existing complete package verification. After copying, every object name must be a lowercase 64-character SHA256, the complete bytes must match that digest, and the declared size must equal the file size. A change during copying fails the post-copy digest check.

Verification requires an exact top-level schema, exact instance fields, a pinned full source-manifest hash, unchanged source config, all 136 runtime bindings, all pinned file sizes, valid complete graph hashes, and exact model.cfg content. It rejects symlinked contracts/object stores/objects, unsafe or unknown hashes, duplicate instances and JSON keys, unbound or unlisted objects, extra top-level files, missing final layers, size lies, same-size corruption, and unknown graph content.

This object-store candidate does not modify `tools/package_model.py` or the native package validator. Existing schema-1/2 behavior remains available. The candidate tool itself only migrates schema 2, and says schema-1 migration is pending; that is an honest limitation rather than a compatibility regression.

## Verification

Executed without model data:

```text
python3 -m unittest tests.test_shape_contract_audit tests.test_dynamic_package tests.test_package -v
Ran 35 tests in 1.023s — OK
```

The test set covers normalized graph field completeness/topology hashes, unknown static manifests, object-store path/hash/size/inventory corruption, runtime-claim mutation, and existing portable package validation. The dynamic-package fixtures mock the production trust registry and expensive source/graph checks; they verify object-store mechanics only. The stored three-package/192-graph audit and 48-record C++ crosscheck provide the separate real graph-text evidence. Neither source is weight or inference evidence.

## Minor and remaining scope

- No new Minor correctness issue was found. The synthetic dynamic-package tests deliberately mock the real package and graph gates; future changes should keep the stored real graph audit or an equivalent explicit integration check alongside them.
- The offline candidate contains only two portable schema-2 instances. The audited schema-1 64x64 package is not migrated into the candidate format.
- Schema-3 native parsing, installed-package behavior, actual shared-weight relocation, complete roughly 22 GiB candidate construction, encoder inventory, posterior/packing identity, and separate encoder/decoder BN epsilon fields remain pending.
- The graph tool can only reproduce the three already reviewed static configurations. It does not generate arbitrary runtime shapes or establish numerical correctness for portrait, extreme, or canary shapes.
- No RoPE/mask values, model tensors, full graph loading, quality thresholds, peak memory, or runtime resource behavior were tested by these commits.

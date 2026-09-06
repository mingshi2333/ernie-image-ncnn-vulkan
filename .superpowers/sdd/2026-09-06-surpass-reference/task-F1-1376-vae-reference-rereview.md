# F1 VAE reference runtime incremental review

Reviewed eff2c34 against frozen v4 plan 5d38b7e0847ba1859b0127ca809c59389540bbf35e7e1552fa13beed4f87a4b0. All305 source hashes independently checked. No torch/model import, GPU, numerical execution or large-model hashing.

Original I1 (no actual worker runtime observation): CLOSED at the execution path. The collector runs before/after model construction, after `_decode`, and after the unchanged wrapper forward. finally calls finish on all exits. Unknown actual files are retained but final authentication fails; changed known files fail; supervisor requires report and authenticated status before declaring completion. Two forwards, resource configuration and numerical calls remain unchanged. Scope honestly excludes transient load/unload between these four boundaries.

New Important I2 — self-consistent incomplete report accepted (`tools/vae_reference_scope.py:92-114`). The validator checks nonempty per-stage files and mapped subset only. With the actual v4 runtime allowlist, retain its smallest single real file in all four phase records; set mapped_files=[]; set union to this same one-file dictionary; keep authentic allowlist/collector/prefix/executable identities. `validate_worker_runtime` accepts this impossible one-file Python/Torch execution inventory. This is a standard-library-only independently reproduced counterexample, not a claim that the honest frozen collector emitted it. Missing-file and missing-phase checks do not close the complete-inventory validation requirement.

Add an explicit required observed coverage contract (collector/entry/interpreter and stable mandatory imported/mapped runtime set, with genuine actual-worker identity), and reject complete-set truncation or empty mappings rather than merely checking provided rows. Add the above deletion/re-signing regression. An arbitrary hostile same-user process is outside this review, but an incomplete report must not pass the independent artifact validator. Keep v4 unchanged if refreezing.

Reproduction construction: choose `name=min(runtime["files"],key=lambda n:runtime["files"][n]["size"])`; set every phase.files and report.files to `{name:runtime["files"][name]}` and every phase.mapped_files to `[]`; retain the accepted identity/status fields; call validate_worker_runtime with actual v4 expected SHA values. Observed result: accepted. No model required.

## I2 closure — 941a127, frozen v5

Independently reviewed frozen plan `bae415829ab5a6a8bf717e0f112085b9389e4f81a701ecaccaa4d2b060aaa2b5`. All307 frozen source SHA values checked. Actual exporter import/input-only probe contains2822 files and137 mappings; mandatory files/maps exactly equal the probe. Parent-only and worker-only sets are empty. Probe PID2198549, parent2198360, start_ticks11360666 matches the preparer Popen identity saved in plan. Probe entry path, entry SHA, collector SHA, runtime and probe manifest SHA all match.

The actual worker command gains no additional forward. The probe path returns before load_vae. Actual execution uses a controller-observed Popen PID/parent/startticks captured before waiting, and independently compares that identity to report and every checkpoint. /proc/stat starttime indexing is correct after separating the parenthesized process name. Four checkpoints retain required files and mappings; a report cannot shrink the mandatory denominator.

Re-ran the exact previous singleton counterexample against actual v5 metadata while supplying correct new identity fields: rejected `Required runtime coverage missing: files=2821 mappings=137`. Independently changed start_ticks: rejected as unauthenticated. Seven worker-runtime unit tests passed. Missing report naturally fails opening, missing/incorrect phases and identities fail before accepting output. Required-set integrity is checked against the actual authenticated probe before launching the model.

Resource contract remains16GiB/swap0/CPU200% with affinity12,14, host floor3GiB and1800s; execute_reference retains the previous50ms host/cgroup polling and session kill on failure. `_decode` and Decode wrapper calls and parameters are unchanged. This review did not import torch, build a model, hash the full model package or execute GPU work.

I2 CLOSED; original I1 remains CLOSED. No remaining Important preparation finding. Root may schedule the concrete frozen run. This is readiness review, not proof of model execution or native/dynamic-shape quality; boundary inventories still explicitly exclude transient load/unload between checkpoints.

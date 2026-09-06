# O2 read/prepare localization and bounded candidates

## Scope and evidence

This is a CPU-only source and artifact analysis. It does not modify runtime math, run a model, or make a formal speed or memory claim. The only timing evidence is the reviewed fixed 64x64, two-CPU, trace-enabled diagnostic in `outputs/execution-metrics-pipeline64-o1-v1`. Its result SHA-256 is `1edfdc2a1834b0943110c958aa60c7b39b1f5c115648fe3cb9b11e0ae6769099`; independent actual review `c11ccb3` found no Important issue.

The CLI host scope was 367,954,317,371 ns. Observed non-overlapping intervals were:

| Phase | Host ns | Samples | Share of CLI host scope |
|---|---:|---:|---:|
| verify | 18,381,651,736 | 1 | 5.00% |
| read_prepare | 320,354,132,790 | 322 | 87.06% |
| compute | 27,458,740,592 | 321 | 7.46% |
| upload | 52,921,153 | 1 | 0.014% |
| final download | 51,055 | 1 | <0.001% |

The classified sum is 99.54% of the CLI host scope. These percentages characterize this diagnostic only. Trace downloads, filesystem cache state, two-CPU scheduling, and the 64x64 graph make them unsuitable as formal S/M results.

## Exact origin of 322 read_prepare samples

`MetricsRecorder::blocks` maps every `BlockSequenceStats::load_seconds` item to `ReadPrepare`. `MetricsRecorder::denoise_step` also maps the sum of each step's input/output head durations to one `ReadPrepare` sample. The final VAE duration is one more composite `ReadPrepare` sample.

The count is therefore exact:

- 25 text block loads: `run_text_blocks` creates a fresh `ncnn::Net`, assigns options, registers layers, calls `load_component`, then destroys the net after that block.
- 288 DiT block loads: 36 streamed blocks x 8 denoise steps. `run_block_sequence` constructs and destroys a fresh net for every block in every step.
- 8 combined head samples: each denoise step calls a newly constructed input-head net and newly constructed output-head net; their two elapsed durations are summed into one sample.
- 1 VAE sample: the whole `decode_vae` call, including its load and compute work.

Total: 25 + 288 + 8 + 1 = 322.

The 321 compute samples are independently explained by 25 text extracts, 288 DiT block extracts/submits, and 8 per-step residual intervals (timestep preparation, packing, modulation preparation, Euler work, finite checks, and other elapsed step work after subtracting heads and block load/compute). This agreement is evidence that the phase accounting is internally consistent; it does not identify a pure disk-read duration.

## What a load sample actually contains

`block_sequence.cpp` starts its load timer before allocating `ncnn::Net`, assigning the full option, clearing session allocators from the net option, setting the Vulkan device, registering custom layers, and calling `load_component`. `component_files.cpp::load_component` calls `load_param_mem` followed by `load_model`.

Pinned ncnn `net.cpp` shows that `load_model` is a composite operation. For each layer it consumes weights through `layer->load_model`, calls `layer->create_pipeline`, calls `layer->upload_model` for Vulkan layers, and may `submit_and_wait` when pending uploads exceed 256 MiB. Consequently the measured 320.354 seconds cannot honestly be labelled disk read, host preparation, pipeline creation, GPU upload, or wait separately. The top-level 52.9 ms upload sample covers only initial latent and conditioning transfer; it excludes weight uploads performed inside every `Net::load_model`.

A `PipelineCache` is already created once in `pipeline.cpp` and passed through the Vulkan option to every streamed net. Shader/pipeline cache reuse therefore already spans all DiT steps, even though each net and its weight allocator are recreated. The current code has no cross-step weight-net cache.

The fixed package sizes show why memory must bound any reuse experiment: input-head weights are 589,447,188 bytes, output-head weights 136,348,172 bytes, and a representative DiT block is 436,241,436 bytes before ncnn/Vulkan expansion. The actual OFF process already reached 10,161,700,864 bytes inside a 10,737,418,240-byte cgroup. Adding persistent weights without a new reviewed cap can exceed that observed margin.

## Required timing split before optimization attribution

A diagnostic-only tagged collector should retain the current non-overlapping total while adding component labels `{text/block-i, dit/step-i/block-j, dit/step-i/input-head, dit/step-i/output-head, vae}` and these nested descriptions:

1. `net_setup_param`: net construction, option/device assignment, custom-layer registration, and `load_param_mem`.
2. `model_load_composite`: the single `load_model` call. It must remain explicitly documented as file consumption + weight decode/unpack + create_pipeline + Vulkan upload + internal waits unless ncnn itself gains separately placed hooks.
3. `extract_compute_composite`: extractor creation/input binding/extract plus the existing explicit submit-and-wait boundary.
4. `net_destroy`: optional lifetime/destruction interval, because streamed weight allocator cleanup may be material.

Nested diagnostic fields must not be added to the top-level phase sum. For failure accounting, a component record becomes complete only after its boundary returns; an in-progress `load_model` remains an incomplete interval with the component/step identity and error. This split can rank components and distinguish repeated heads from repeated blocks without claiming unavailable GPU/kernel time.

## Candidate 1: retain only the DiT output-head net across denoise steps

The smallest reuse candidate is a request-local, Vulkan-device-bound output-head session constructed once before the step loop and reused through a fresh extractor on each of eight steps. It replaces eight load/create/upload cycles with one, saving seven composite loads of the 136,348,172-byte output-head weights. It leaves block order, input head, scheduler, Euler update, tensor layout, precision, pipeline cache, and every arithmetic operation unchanged.

This is preferred over retaining both heads or arbitrary blocks for the first experiment because the actual OFF run had only about 576 MiB between observed process peak and its 10 GiB cgroup limit. The output head is substantially smaller than the 589 MB input head or a 436 MB block before runtime expansion. Even this candidate needs a freshly reviewed memory cap; source size is not an upper bound on Vulkan allocation.

Implementation boundary: add an internal request-owned loaded-head object in `dit`/`denoiser`; never use a global or cross-request cache. Its net must use the same device, `Option`, pipeline cache, and allocator policy. Destroy it before the request GPU context and allocators. Keep the old streaming path as the default until an opt-in diagnostic proves the contract.

Validation: same frozen source/package/input except the opt-in, trace disabled for a timing pilot and enabled for numerical verification. Require all 27 fixed trace files and PNG byte-identical to baseline, exact per-step predictions and Euler outputs, unchanged token IDs and constants, zero live allocations at teardown, device identity match, no OOM/swap, and a new tagged count of one output-head load plus eight output-head extracts. Measure at least paired AB/BA repetitions before any performance conclusion. A failure or missing tagged interval is ineligible.

## Candidate 2: opt-in mapped loading for streamed model files

A second small candidate is request-local `Option::use_mapped_model_loading=true` for the existing streamed nets. Pinned ncnn's path maps the model file and calls the same model loader; net lifetime retains the mapping. It does not intentionally alter graph operations, weights, precision, scheduler, or execution order. It may reduce ordinary file copying and allocator churn, but it does not remove weight decode, pipeline creation, Vulkan upload, or internal waits, so benefit is uncertain and must be measured.

Implementation boundary: expose it only as an internal diagnostic option initially, apply the same value to text/DiT/head/VAE loading under test, and keep the default false. Do not combine it with resident-net caching in the first experiment. Record mapped-open failure/fallback explicitly: the current ncnn implementation may fall back to regular file loading, and an unobserved fallback cannot be reported as a mapped result.

Validation: require an explicit `mapped_loading_used` observation per component rather than inferring it from the requested option; preserve the same full tensor/PNG byte denominator, component load counts, device/precision identity, allocation cleanup, and guard evidence. Compare opt-in versus default with paired AB/BA runs only after numerical equality. Report the composite model-load interval honestly; no result may be described as reduced disk time without lower-level evidence.

## Decision boundary

Candidate 1 is the first implementation choice if a no-model contract can establish request ownership and safe destruction, followed by one bounded actual memory/numerical experiment. Candidate 2 is lower-risk in code size but has uncertain benefit because Vulkan upload and pipeline work remain inside `load_model`. Neither candidate justifies changing defaults from this single diagnostic. Broader caching, asynchronous prefetch, keeping multiple 436 MB blocks resident, ncnn modifications, and formal performance runs remain outside this slice.

# Compact temporary ModelBin buffer: opt-in implementation

The pinned ncnn reader allocates twice the required temporary space when it
cannot reference FP16/BF16 input directly. The local opt-in correction halves
that array while preserving the aligned read length and decoded FP32 bits.
This is a reader allocation result, not a whole-process memory reduction,
complete-model result, or image-generation speedup.

## Implementation and identity

- Project base: `d3e14a8`; the source inventory records the local implementation
  and build/CI files accompanying this report.
- Original ncnn: `6a1bf000f363714839a36793addc8c879d3d899e`, clean and unchanged.
- Original `src/modelbin.cpp` SHA256:
  `eefc538407f88ca9cb3c1c3bde30eb4aa764b1d1c5f4bba9ab9cdd9119c3cb78`.
- Derived file SHA256:
  `90adcd77b0a3c2cf49142d65f6a8dc74dadc26ef7460c162c2b21721234f7a69`.
- Exact two-line change: [model-reader.patch](model-reader.patch).

`align_data_size` is a byte count. The original passes it as an element count
to `vector<unsigned short>::resize`. The correction divides by
`sizeof(unsigned short)` in the FP16 and BF16 ordinary-read branches. The read
still consumes the same 4-byte-aligned payload and decodes only the declared
number of values. Successful reference reads and FP32 on-disk data are unchanged.
FP32 inference can still load BF16-encoded on-disk weights.

`cmake/ErnieModelReader.cmake` authenticates the original file, generates a copy
under the selected build directory, and replaces exactly one ncnn compilation
unit. The option `ERNIE_EXPERIMENT_COMPACT_MODEL_READER` defaults to `OFF`.
Changing even one byte of the original is rejected at configuration; the
expected rejection and exact source comparison are retained in
[source-checks.json](source-checks.json).

## Actual validation

The native contract directly invokes `ncnn::ModelBinFromDataReader`. A generated
non-reference reader observes the temporary vector's actual allocation request
through the executable's global allocator; this is not an RSS estimate. The
reference branch is also executed using `DataReaderFromMemory`. Separate
component-loading contracts exercise real file/mapped loading and ownership.

| Local Release build | Experiment | CPU-executed CTests | Largest aligned payload | Observed temporary allocation |
|---|---|---:|---:|---:|
| CPU | OFF | 3/3 pass | 8,388,612 B | 16,777,224 B |
| CPU | ON | 3/3 pass | 8,388,612 B | 8,388,612 B |
| Vulkan linked | OFF | 3/3 pass | 8,388,612 B | 16,777,224 B |
| Vulkan linked | ON | 3/3 pass | 8,388,612 B | 8,388,612 B |

The three tests are `model_reader_cpu`, `model_loading_cpu`, and
`component_files_contract_cpu`: twelve executions of three tests, with no
failure or skip. Both builds use the recorded local Clang toolchain. No test in
this table submits GPU work. Each reader test checks both storage formats at
1, 2, 3, 17, 2,049, 65,537 and 4,194,305 values; positive and negative zero,
ordinary values and a subnormal are checked by exact FP32 bit pattern. Odd-size
padding must be consumed but not decoded. Truncated payloads and tags must fail;
their expected error messages remain in the logs.

The added Linux CI configuration enables the experiment in one CPU job, while
the existing CPU/Vulkan jobs retain `OFF`. The local workflow parser verifies
all three configurations. **Remote CI has not executed this change.**

Builds ran with two assigned CPU cores, a 200% CPU quota, a 4 GiB cgroup limit
and no cgroup swap. The concurrently running image comparison uses a different
CPU assignment and its own frozen binary/source copies. These builds do not
alter any running trial or establish isolated performance measurements.

## Candidate preservation and default restoration

Candidate binaries and reader probes are retained outside Git under
`outputs/compact-model-reader-v1/candidate/{cpu,vulkan}/`.
[candidate-identity.json](candidate-identity.json) records hashes and complete
configuration caches. The Vulkan candidate executable SHA256 is
`b15ccce8e92962b041df6581926dd7c06f1f75b65b1b2cbb949f927be71ca05a`.

Both standard local build trees were restored to `OFF` and rebuilt. Their ncnn
archives match the saved originals byte-for-byte; the CPU reader probe also
matches its original. The Vulkan CLI is restored exactly to
`ff2f934f317a602f3d098bf6d74cd7c2e2a6de725ed8fcf04d8d1e2bcbaa3e1c`.
These checks are in [restored-baselines.json](restored-baselines.json); restoring
identical artifacts did not add another claimed test suite.

## Remaining work

The candidate still needs serial complete-model validation after the active
[four-configuration comparison](../memory-grid-protocol/README.md) finishes.
That existing comparison uses the original reader throughout; its results
cannot be attributed to this patch. A new candidate comparison must bind its
own binary and inputs before execution. No cache/loading default is changed.

This change does not add activation spilling, failed Vulkan-command recovery,
prefetch, or general RAM capacity guarantees. It does not close O2, formal peer
speed/memory acceptance, broad quality/shape validation, platform delivery or
the full surpass-reference objective. No push or release was performed.

Local reproduction commands and applicability are in
[REPRODUCE-COMPONENTS.md](../../../docs/REPRODUCE-COMPONENTS.md#普通读取的临时权重缓冲区实验).
The source inventory and record inventory bind the files and logs accompanying
this report; generated binaries and model data are intentionally outside Git.

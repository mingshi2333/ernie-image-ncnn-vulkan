# F3 integration fixes: independent rereview

Review target: the uncommitted fixes in `cli/image_io.cpp`, `tests/test_image_io.cpp`, `src/gpu_context.*`, `src/pipeline.cpp`, `src/runtime_info.cpp`, and `tests/test_gpu_context.cpp`, relative to the four Important findings in `task-F3-integration-review.md`. This was source review plus small standalone CPU reproducers. It did not load a model or perform GPU inference.

## Verdict

No Critical or Important finding remains in the reviewed fix slice. All four earlier Important findings are closed by code and focused regression evidence. The full F3 feature remains subject to the separate image-to-image integration and model-run gates; this review only closes the named I/O and GPU-lifetime defects.

## Closed findings

### Important 1: concurrent diagnosis could destroy another live GPU instance — closed

`src/gpu_context.cpp` now owns the one process-local ncnn lifetime guard used by both generation and diagnosis. Constructor/destructor transitions are serialized; `context_users` holds the instance until the final library user exits. The first user records whether the instance was created by this library. If an external instance already exists it is borrowed and never destroyed by the library. Constructor validation failure destroys only a newly created, still-unshared instance. `runtime_info.cpp` retains its context while copying the complete device list, and `pipeline.cpp` uses the same class.

`tests/test_gpu_context.cpp` covers nested diagnosis during a live generation-like owner, two concurrent diagnoses, failed nested device validation, final owned-instance cleanup, and preservation of an externally created instance. The external-owner contract is stated in the header: that owner must retain the borrowed instance until library calls return. Arbitrary unsynchronized destruction by external ncnn users is outside a library-only reference count and is not represented as supported.

### Important 2: corrupt JPEG null failure reason caused SIGSEGV — closed

All stb header/decode errors now go through `decode_error()`, which uses a stable fallback when `stbi_failure_reason()` returns null. The original corrupt-JPEG standalone reproducer now reports `Image decode failed: invalid or truncated image` and exits normally rather than reaching `strlen(nullptr)`.

### Important 3: late write failure was reported as success — closed

Encoding is completed in memory and `save_new()` uses exclusive `O_CREAT|O_EXCL` creation, an EINTR-aware write loop, and an explicitly checked `close`. It reports failure when either writing or close fails. The original isolated `RLIMIT_FSIZE=0`, ignored-`SIGXFSZ` reproducer now reports `Image write failed`. The exclusive open also removes the earlier exists-check/time-of-check race and the regression test verifies that a rejected second write does not alter the existing image.

### Important 4: header-only BMP/TGA produced fabricated pixels — closed

The stb formats now decode through bounded callbacks. A legal short initial prefetch returns the available bytes without marking truncation. A later read at EOF, a negative request, or an out-of-range skip marks the input truncated; a nominal stb decode is rejected when that flag is set. Thus this does not require the decoder to consume trailing metadata or every physical byte, but it does reject a decoder attempt to obtain absent payload. The original 54-byte BMP and 18-byte TGA reproducers now both fail with `truncated image payload`. A JPEG with its final ten bytes removed is also rejected.

## Independent reproduction

The original review program was rebuilt directly against the current `cli/image_io.cpp` and run in two new temporary directories. Output:

```text
.bmp rejected Image decode failed: truncated image payload
.tga rejected Image decode failed: truncated image payload
write rejected Image write failed
.jpg rejected Image decode failed: invalid or truncated image
write rejected Image write failed
```

This is a small CPU-only reproduction. The maintained `image_io_contract` covers ordinary lossless PNG/BMP/TGA roundtrips, JPEG-95 error, grayscale, alpha backgrounds, Unicode paths, exclusive output, truncation, late write failure, corrupt headers and oversized declarations.

Root's completed build session `12061` exited successfully. The subsequently executed focused CTest set `request_validation`, `gpu_context`, `image_io`, and `CLI` passed all four tests in `6.15s`. In particular, the compiled Vulkan-enabled `gpu_context` contract exercised real device enumeration plus nested and concurrent users, failed construction, external-instance borrowing, and final owned-instance release; it did not submit model compute.

## Minor observation

A failed low-level write can leave the newly and exclusively created path as a partial or empty file. The operation returns an error and never reports success, so this does not reopen the original finding. Callers must remove that failed artifact or choose a new output path before retrying. Automatic removal would be a cleanup improvement, provided it never removes a pre-existing file.

## Boundaries

- Device enumeration is allowed by this review; no Vulkan compute or model inference was performed.
- Source inspection establishes the shared lifetime policy. Actual hardware-dependent device enumeration remains represented by the dedicated small contract test rather than a complete generation run.
- This report does not mark F3 complete and does not review the concurrently changing dynamic package, text-down, tokenizer schema, or image encoder implementation.

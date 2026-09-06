# F3 early image I/O component report

This is the independently testable CLI image-codec portion of P3 Task F3. It does not modify or connect pipeline, request, option, or main-program behavior, and therefore does not claim F3 is complete. No model or GPU process was run.

## Implemented component

- Preserved `write_png(path, image)` and added `read_image(path, background)` plus `write_image(path, image)` in `cli/image_io.*`.
- PNG decode/encode continues through the system libpng simplified API. JPEG, BMP, and TGA use vendored upstream `nothings/stb` at commit `2c980bb59875b0d32144a71867fbdebb2f77cd20`.
- All codecs operate on in-memory bytes and C++ filesystem streams. The codec libraries do not receive a lossy filename conversion, so the tested Linux UTF-8/Chinese path works through `std::filesystem::path`.
- Decode always returns interleaved 8-bit RGB. Grayscale is replicated across RGB. Alpha uses exact integer source-over composition; the default background is white and callers can provide an explicit RGB background.
- JPEG output is fixed at quality 95. PNG/BMP/TGA are lossless for RGB inputs.
- Extension matching is case-insensitive and limited to PNG, JPG/JPEG, BMP and TGA. Outputs refuse overwrite.
- File size, dimensions, pixel count, channel multiplication and RGB buffer size are checked before target allocation/encoding. Empty, inaccessible, corrupt, oversized and huge-declaration inputs fail with exceptions.
- Codec dependencies remain in `cli/image_io.cpp`; neither the public pipeline header nor the runtime library imports libpng/stb.

## Vendored source identity

Primary source: `https://github.com/nothings/stb` revision `2c980bb59875b0d32144a71867fbdebb2f77cd20`.

```text
594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3  third_party/stb/stb_image.h
cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05  third_party/stb/stb_image_write.h
bebfe904b14301657e4e5d655c811d51fd31b97c455b9cc2d8600d6bac6cff63  third_party/stb/LICENSE
```

The revision, role and all three hashes are recorded as an additional `stb` entry in `sources.lock.json`; existing dependency pins were not changed.

## Actual tests

`tests/test_image_io.cpp` performs real codec operations on generated pixels:

- PNG, BMP and TGA encode/decode roundtrips must be pixel-identical.
- JPEG quality-95 roundtrip must retain dimensions and mean absolute RGB error below 8.
- Gray PNG must replicate channels; RGBA PNG must match default white and explicit black compositing.
- Chinese filename roundtrip must succeed on this Linux host.
- Existing output, corrupt JPEG, structurally valid BMP with declared 50000 x 50000 dimensions, overflowing RGB dimensions, and misuse of `write_png` must be rejected.

Commands and results:

```text
cmake -S . -B build-dev -DERNIE_BUILD_GENERATOR=ON
cmake --build build-dev --target ernie-image-io-contract -j2
[100%] Built target ernie-image-io-contract

ctest --test-dir build-dev -R image_io_contract --output-on-failure
1/1 Test #9: image_io_contract ... Passed
100% tests passed, 0 tests failed out of 1

cmake --build build-dev --target ernie-image -j2
[100%] Built target ernie-image

python -m json.tool sources.lock.json
exit 0

git diff --check
exit 0
```

## Pending F3 integration

The new APIs are not called by `cli/main.cpp`; PNG remains the current application output until root integrates format options. Input-image request plumbing, `--background`, resize stretch/fit/crop and trace metadata, strength/request validation, full CLI help and real Windows UTF-8 boundary testing remain pending. The current Linux Chinese-path test is not evidence for Windows behavior.

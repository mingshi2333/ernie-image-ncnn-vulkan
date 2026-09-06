# Local delivery drafts

No public release or model download endpoint has been established by these tools.
An archive produced by `build_release.py` is a local review draft. Its manifest
always records `distributable: false`, `published: false`, and an unverified
release platform. Successful archive creation does not approve redistribution.

Use Python 3.11 or newer to assemble an existing installation:

```sh
python3 tools/build_release.py --install /path/to/install --output /new/draft \
  --platform linux-x86_64 --source /path/to/frozen-source \
  --build-evidence /path/to/build-evidence \
  --install-evidence /path/to/installation/result.json \
  --cargo-registry /path/to/cargo/registry/src \
  --ncnn-source /path/to/pinned/ncnn
```

The default candidate contains the CLI and its installed project license,
locked source references and running instructions. `--include-sdk` explicitly
adds installed public headers, static archives and CMake/pkg-config metadata.
It never recursively copies the source checkout, Python environment, model
directories or arbitrary installation files. The output directory must be new;
failed drafts retain `FAILED.txt` for investigation. The original installation
is read-only input.

The output includes:

- `ernie-runtime/`: selected files, collected notice texts, `release.json` and a
  prominent draft notice.
- `files.json`: full SHA256 and size of every payload file, including the release
  manifest. It does not include itself or the enclosing archive.
- `ernie-runtime-draft.tar.gz` and `SHA256SUMS`: deterministic archive metadata and
  checksums for the archive and file inventory.

ELF dependencies are inspected with `readelf`, never by executing the input.
System shared libraries are not bundled. Available `ldconfig`/RPM records and
license files are recorded as host candidates, not as proof of the actual
loader resolution or permission to redistribute those libraries. Transitive
dynamic dependency resolution remains unverified.

Local project and stb notices are collected from the provided source. ncnn
and glslang notices come from the exact Git objects named by sources.lock.json
and its glslang submodule pin; --ncnn-source can supply a separate local checkout.
Rust license texts are read directly from local .crate archives whose full
SHA256 matches Cargo.lock. An extracted registry cache is also supported when
its package checksum and individual Cargo.toml/notice checksums match. The lock inventory is conservative:
it can include build and optional dependencies. Neither Cargo metadata nor RPM
license expressions establish a complete linked notice closure or legal
redistribution basis. Missing texts, unknown origins, source/build mismatches
and absent platform evidence remain explicit blockers. Owners must review the
actual release contents and the applicable terms before any distribution.

## Downloading model bytes separately

`tools/download_model.py` and the adjacent `release_manifest.py` use only the
Python standard library. A separately supplied schema-1 download manifest must
identify an immutable revision, graph schema, required capabilities, conversion
source and script checksum, license source/notice, and each file's safe relative
path, immutable HTTPS URL, byte size and SHA256. No production manifest with
invented release URLs is supplied.

```sh
python3 tools/download_model.py --manifest /path/to/download.json --output /path/to/model
```

The downloader displays the total size and destination, authenticates existing
completed files, resumes partial responses, and publishes a final name only
after the complete file hash passes. It fails on changed object versions,
invalid ranges, truncated or corrupt data, unsafe paths and unauthenticated
existing targets. A server that ignores Range causes a full restart. Disk
checks conservatively allow that restart. Atomic no-overwrite completion needs
a filesystem supporting hard links. A lock left by a force-killed process
requires checking that the old process is gone before manual cleanup.

Download integrity is separate from model semantics. After bytes are obtained,
run the installed native model/package verifier and the relevant PE verifier.
The downloader does not establish model quality, feature support, provenance
beyond the supplied authenticated manifest, or redistribution permission.

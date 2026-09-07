"""Local MinGW/Wine SDK diagnostic; does not certify native Windows support."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path.cwd()
REPOSITORY = ROOT.parents[1]
RECORDS = ROOT / "outputs/windows-cpu-v1"
BUILD = ROOT / "build-windows-cpu-v1"
OUT = Path("/var/tmp/ernie-windows-sdk-v1")
OUT.mkdir(exist_ok=False)
PREFIX = OUT / "安装 original"
MOVED = OUT / "移动 installation"
CONSUMER = OUT / "consumer"
shutil.copytree(ROOT / "tests/install", CONSUMER)
shutil.copyfile(RECORDS / "toolchain.cmake", OUT / "toolchain.cmake")
env = os.environ.copy()
for key in ("DISPLAY", "WAYLAND_DISPLAY", "CMAKE_PREFIX_PATH", "CMAKE_MODULE_PATH",
            "Ernie_DIR", "ncnn_DIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBRARY_PATH",
            "CPATH", "CPLUS_INCLUDE_PATH", "VIRTUAL_ENV", "PYTHONPATH"):
    env.pop(key, None)
env.update(WINEPREFIX=str(OUT / "wine-prefix"), WINEARCH="win64", WINEDEBUG="-all",
           WINEDLLOVERRIDES="winemenubuilder.exe=d", OMP_NUM_THREADS="2",
           WINEPATH="Z:" + str(MOVED / "bin").replace("/", "\\"), TMPDIR=str(OUT / "temporary"))
Path(env["TMPDIR"]).mkdir()
result = {"schema_version": 1, "status": "running", "commands": [],
          "scope": "Cross-compiled Windows CPU SDK and CLI under Wine; model-free, no native Windows or GPU claim",
          "source_hidden": False, "network_disabled": False}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save():
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def run(args, name, wrapper=(), expected=0):
    command = [*wrapper, *map(str, args)]
    start = time.monotonic()
    with (OUT / (name + ".log")).open("wb") as log:
        completed = subprocess.run(command, env=env, cwd=OUT, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=600)
    result["commands"].append({"name": name, "argv": command, "return_code": completed.returncode,
                               "expected": expected, "seconds": time.monotonic() - start})
    save()
    if completed.returncode != expected:
        raise RuntimeError(f"{name} returned {completed.returncode}, expected {expected}")
    return (OUT / (name + ".log")).read_text(errors="replace")


try:
    run(["cmake", "--install", BUILD, "--prefix", PREFIX], "install")
    PREFIX.rename(MOVED)
    for path in MOVED.rglob("*.cmake"):
        content = path.read_text()
        if any(str(p) in content for p in (ROOT, BUILD, PREFIX, REPOSITORY)):
            raise RuntimeError(f"Installed config retains a source/build/prefix path: {path}")
    runtimes = json.loads((RECORDS / "windows-imports.json").read_text())["runtime_copies"]
    for name, entry in runtimes.items():
        source = Path(entry["source"])
        if source.stat().st_size != entry["bytes"] or sha(source) != entry["sha256"]:
            raise RuntimeError(f"Runtime DLL identity changed: {source}")
        shutil.copyfile(source, MOVED / "bin" / name)
    result["runtime_dlls"] = runtimes
    # CMake cross-rooting still needs to see this installed target prefix.
    with (OUT / "toolchain.cmake").open("a") as toolchain:
        toolchain.write(f'list(APPEND CMAKE_FIND_ROOT_PATH "{MOVED}")\n')
    wrapper = ["bwrap", "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/",
               "--bind", str(OUT), str(OUT), "--tmpfs", str(REPOSITORY),
               "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--chdir", str(OUT), "--"]
    run(["/usr/bin/python3", "-c", "from pathlib import Path; import socket; "
         f"assert not list(Path({str(REPOSITORY)!r}).iterdir()); "
         "assert socket.if_nameindex() == [(1, 'lo')]; "
         "print('Source hidden; private network contains loopback only')"],
        "isolation-preflight", wrapper)
    result.update(source_hidden=True, network_disabled=True, hidden_paths=[str(REPOSITORY)])
    foreign = OUT / "foreign-ncnn"
    foreign.mkdir()
    (foreign / "ncnnConfig.cmake").write_text('message(FATAL_ERROR "Foreign ncnn cache was used")\n')
    consumer_build = OUT / "consumer-build"
    run(["cmake", "-S", CONSUMER, "-B", consumer_build, "-G", "Ninja",
         f"-DCMAKE_TOOLCHAIN_FILE={OUT / 'toolchain.cmake'}", f"-DCMAKE_PREFIX_PATH={MOVED}",
         f"-Dncnn_DIR={foreign}", "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
         "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], "consumer-configure", wrapper)
    if f"ncnn_DIR:UNINITIALIZED={foreign}" not in (consumer_build / "CMakeCache.txt").read_text():
        raise RuntimeError("Ernie changed the caller's ncnn_DIR cache")
    run(["cmake", "--build", consumer_build, "--parallel", "2"], "consumer-build", wrapper)
    run(["/usr/bin/wine", "wineboot", "-u"], "wineboot", wrapper)
    run(["ctest", "--test-dir", consumer_build, "--output-on-failure",
         "--output-junit", OUT / "consumer-ctest.xml"], "consumer-test", wrapper)
    executable = MOVED / "bin/ernie-image.exe"
    run(["/usr/bin/wine", executable, "--help"], "installed-help", wrapper)
    diagnosis = run(["/usr/bin/wine", executable, "--diagnose"], "installed-diagnose", wrapper)
    if "vulkan_compiled=false" not in diagnosis:
        raise RuntimeError("Installed CPU executable reported unexpected Vulkan support")
    damaged = OUT / "损坏 model"
    damaged.mkdir()
    (damaged / "manifest.json").write_text('{"schema_version":3}')
    rejection = run(["/usr/bin/wine", executable, "--model", damaged, "--verify-model"],
                    "installed-corrupt-package", wrapper, expected=1)
    if "Unexpected schema-3 fields" not in rejection:
        raise RuntimeError("Installed CLI did not reach the expected Unicode-path package validation")
    result["installed_files"] = [{"path": str(p.relative_to(MOVED)), "bytes": p.stat().st_size,
                                  "sha256": sha(p)} for p in sorted(MOVED.rglob("*")) if p.is_file()]
    result["status"] = "passed"
except BaseException as error:
    result.update(status="failed", error=str(error))
    raise
finally:
    save()

"""Install, move the prefix, then build a separate standard C++ application.

This uses actual libraries. It never downloads weights or generates an image.
Linux --isolate-source hides the source/build checkout and disables networking
for consumer configuration, linking, CLI checks and execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_install(build, output, config="Release", isolate=False):
    build, output = Path(build).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prefix = output / "安装 original"
    moved = output / "移动 installation"
    consumer = output / "consumer"
    shutil.copytree(PROJECT / "tests/install", consumer)
    temporary = output / "temporary"
    temporary.mkdir()
    environment = os.environ.copy()
    for key in ("CMAKE_PREFIX_PATH", "CMAKE_MODULE_PATH", "Ernie_DIR", "ncnn_DIR",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
                "CPLUS_INCLUDE_PATH", "VIRTUAL_ENV", "PYTHONPATH"):
        environment.pop(key, None)
    environment["TMPDIR"] = str(temporary)
    commands = []
    result = {"schema_version": 1, "status": "running", "source_isolated": False,
              "network_disabled": False, "scope": "installed model-free C++ API and CLI",
              "commands": commands}

    def run(command, name, *, wrapper=(), expected=0):
        actual = [*wrapper, *map(str, command)]
        completed = subprocess.run(actual, cwd=output, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        (output / (name + ".log")).write_text(completed.stdout)
        commands.append({"name": name, "argv": actual, "exit_code": completed.returncode})
        if completed.returncode != expected:
            raise RuntimeError(f"{name} returned {completed.returncode}: {completed.stdout[-4000:]}")
        return completed.stdout

    try:
        run(["cmake", "--install", build, "--config", config, "--prefix", prefix], "install")
        prefix.rename(moved)
        configs = list(moved.rglob("*.cmake"))
        if not configs or not (moved / "include/ernie/pipeline.h").is_file():
            raise ValueError("No installed public SDK; configure ERNIE_INSTALL_SDK=ON")
        for path in configs:
            content = path.read_text()
            for forbidden in (str(PROJECT), str(build), str(prefix)):
                if forbidden in content:
                    raise ValueError(f"Installed CMake file still refers to original paths: {path}")
        wrapper = []
        if isolate:
            if sys.platform != "linux" or not shutil.which("bwrap"):
                raise RuntimeError("Source isolation requires Linux bubblewrap")
            # Hide the repository containing shared models, ncnn and this worktree.
            git_common = subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse",
                "--path-format=absolute", "--git-common-dir"], text=True).strip()
            repository = Path(git_common).parent.resolve()
            hidden = sorted({repository, PROJECT, build}, key=lambda p: len(p.parts))
            roots = []
            for path in hidden:
                if path == output or path in output.parents:
                    raise ValueError("Isolated install output must be outside source/build directories")
                if not any(root == path or root in path.parents for root in roots):
                    roots.append(path)
            wrapper = ["bwrap", "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/",
                       "--bind", str(output), str(output), "--dev-bind", "/dev", "/dev",
                       "--proc", "/proc", "--chdir", str(output)]
            for path in roots:
                wrapper += ["--tmpfs", str(path)]
            wrapper += ["--"]
            run(["cmake", "-E", "capabilities"], "isolation-preflight", wrapper=wrapper)
            result.update(source_isolated=True, network_disabled=True, hidden_paths=list(map(str, roots)))
        consumer_build = output / "consumer-build"
        # Reuse the configured compiler family so its standard/OpenMP runtimes match.
        cache = (build / "CMakeCache.txt").read_text().splitlines()
        compiler = next(line.split("=", 1)[1] for line in cache if line.startswith("CMAKE_CXX_COMPILER:"))
        configure = ["cmake", "-S", consumer, "-B", consumer_build,
                     f"-DCMAKE_PREFIX_PATH={moved}", f"-DCMAKE_CXX_COMPILER={compiler}",
                     "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
                     "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_BUILD_TYPE=Release"]
        foreign_ncnn = output / "foreign-ncnn"
        foreign_ncnn.mkdir()
        (foreign_ncnn / "ncnnConfig.cmake").write_text('message(FATAL_ERROR "Foreign ncnn cache was used")\n')
        configure.append(f"-Dncnn_DIR={foreign_ncnn}")
        if shutil.which("ninja"):
            configure += ["-G", "Ninja"]
        run(configure, "consumer-configure", wrapper=wrapper)
        consumer_cache = (consumer_build / "CMakeCache.txt").read_text()
        if f"ncnn_DIR:UNINITIALIZED={foreign_ncnn}" not in consumer_cache:
            raise ValueError("Ernie modified the caller's cached ncnn_DIR")
        run(["cmake", "--build", consumer_build, "--config", config, "--parallel", "2"],
            "consumer-build", wrapper=wrapper)
        run(["ctest", "--test-dir", consumer_build, "-C", config, "--output-on-failure"],
            "consumer-test", wrapper=wrapper)
        executable = moved / "bin" / ("ernie-image.exe" if os.name == "nt" else "ernie-image")
        run([executable, "--help"], "installed-help", wrapper=wrapper)
        run([executable, "--diagnose"], "installed-diagnose", wrapper=wrapper)
        damaged = output / "损坏 model"
        damaged.mkdir()
        (damaged / "manifest.json").write_text('{"schema_version":3}')
        rejection = run([executable, "--model", damaged, "--verify-model"],
                        "installed-corrupt-package", wrapper=wrapper, expected=1)
        if "Unexpected schema-3 fields" not in rejection:
            raise ValueError("Installed CLI failed without the expected package validation error")
        result["status"] = "passed"
        result["installed_files"] = [{"path": str(p.relative_to(moved)), "bytes": p.stat().st_size,
            "sha256": file_sha256(p)}
            for p in sorted(moved.rglob("*")) if p.is_file()]
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class InstalledConsumerTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("ERNIE_INSTALL_BUILD"), "Requires a built ERNIE_INSTALL_SDK=ON tree")
    def test_relocated_installation(self):
        with tempfile.TemporaryDirectory(prefix="ernie-install-") as temp:
            result = check_install(os.environ["ERNIE_INSTALL_BUILD"], Path(temp) / "check",
                                   config=os.environ.get("ERNIE_INSTALL_CONFIG") or "Release")
            self.assertEqual(result["status"], "passed")


if __name__ == "__main__" and "--build" in sys.argv:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="Release")
    parser.add_argument("--isolate-source", action="store_true")
    args = parser.parse_args()
    check_install(args.build, args.output, args.config, args.isolate_source)
else:
    if __name__ == "__main__":
        unittest.main()

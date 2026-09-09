"""Audit selected v2 FP32, v3 BF16 and v4 default FP16 frozen executions.

Run only after all model jobs and frozen compare_full.py runs have ended.
This rereads large model bindings once per unique path, authenticates source
snapshots against committed Git objects, and independently recalculates output
identities and the original numerical gates. It does not run models or tests.
"""

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tarfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("execution-audit.json")
NCNN = "3b7bdba7fc8aea8fd46779533eee027df77c639d"
PINS = {
    "v2": {
        "source": "5b57e9dee4283c5c24fcfdb4bb03349017424253",
        "plan": "b794844d3877532e9d9d8eecd813503f089e883be7bf35425b544c9d38e323f3",
        "binary": "8d8c3216b0d1addf401eeab3d45eddb4aa64b973a05cbf987f20cdb49da49ecd",
        "source_files": 357, "bindings": 158, "derived": 4,
        "selected": ["normal-fp32", "mixed-prefetch-fp32"],
        "batch_status": "validation_failed",
    },
    "v3": {
        "source": "7296bfeba2c809d54a7d6214b0f08dee861ad1f8",
        "plan": "82c096eacb80658a3bd2637218708b1a5ae0381e067823311160cd3d860401c3",
        "binary": "3202547804114b542a2785a1466bfc117e5d75830bf0633daf7b58cb741910da",
        "source_files": 358, "bindings": 158, "derived": 5,
        "selected": ["bf16-gemm-bf16"], "batch_status": "complete",
    },
    "v4": {
        "source": "7296bfeba2c809d54a7d6214b0f08dee861ad1f8",
        "plan": "80ba498048f0c6a9e29a4fe13a3a908d66f70f64f860c3192fc13c51b519b1cd",
        "binary": "3202547804114b542a2785a1466bfc117e5d75830bf0633daf7b58cb741910da",
        "source_files": 358, "bindings": 159, "derived": 5,
        "selected": ["default-fp16"], "batch_status": "complete",
    },
}
GATES = {
    "fp32": {"nrmse": .003, "global_rtol": .01, "atol": .0002,
             "pixel_mae": .1, "pixel_max": 2},
    "fp16": {"nrmse": .15, "global_rtol": .25, "atol": .03,
             "pixel_mae": 12, "pixel_max": 80},
    "bf16": {"nrmse": .15, "global_rtol": .25, "atol": .03,
             "pixel_mae": 12, "pixel_max": 80},
    "conditioning": {"nrmse": .0002, "global_rtol": .0002, "atol": .0002},
}
IDENTITIES = {}
STAMPS = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def identity(path):
    path = Path(path).absolute()
    before = stamp(path)
    if path in IDENTITIES:
        require(before == STAMPS[path], f"File changed during audit: {path}")
        return IDENTITIES[path]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(stamp(path) == before, f"File changed while hashing: {path}")
    STAMPS[path] = before
    IDENTITIES[path] = {"sha256": digest, "bytes": before[2]}
    return IDENTITIES[path]


def read_json(path):
    identity(path)
    return json.loads(Path(path).read_text())


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def validation(path):
    identity(path)
    text = Path(path).read_text()
    return {
        "vuid_mentions": len(re.findall(r"VUID-[A-Za-z0-9_-]+", text)),
        "vuid_lines": sum("VUID-" in line for line in text.splitlines()),
        "validation_error_mentions": len(re.findall("Validation Error", text, re.IGNORECASE)),
    }


def assert_no_running_models(bases):
    watched = {str(base / "frozen-bin/ernie-image") for base in bases}
    watched.update(str(base / "run_full.py") for base in bases)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not watched.intersection(argv), f"Frozen model/supervisor is still running: {process.name}")


def verify_sources(base, pin):
    source = read_json(base / "source-bindings.json")
    files = source["files"]
    require(source["head"] == pin["source"] and len(files) == pin["source_files"], "Source identity/count differs")
    with tarfile.open(base / "source-snapshot.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len(files) and {x.name for x in members} == set(files), "Source archive membership differs")
        for member in members:
            require(member.isfile(), f"Non-file source archive entry: {member.name}")
            payload = archive.extractfile(member).read()
            expected = files[member.name]
            require({"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)} == expected,
                    f"Source archive digest differs: {member.name}")
    # One cat-file process authenticates every archived source to the pinned
    # commit. The live worktree may already contain newer documentation.
    names = list(files)
    requests = "".join(f"{pin['source']}:{name}\n" for name in names).encode()
    data = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input=requests,
                          check=True, stdout=subprocess.PIPE).stdout
    stream = io.BytesIO(data)
    for name in names:
        header = stream.readline().split()
        require(len(header) == 3 and header[1] == b"blob", f"Missing committed source: {name}")
        payload = stream.read(int(header[2]))
        require(stream.read(1) == b"\n", "Invalid cat-file response")
        require({"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)} == files[name],
                f"Committed source digest differs: {name}")
    require(stream.read() == b"", "Unexpected committed-source response")
    require(git("ls-tree", pin["source"], "third_party/ncnn").decode().split()[2] == NCNN,
            "Pinned ncnn gitlink differs")
    return source


def verify_derived(base, pin):
    derived = read_json(base / "derived-compiled-units.json")["files"]
    commands = read_json(base / "compile_commands.json")
    require(len(derived) == pin["derived"], "Derived compilation unit count differs")
    for name, record in derived.items():
        matches = [item for item in commands if item["file"].endswith("/" + name)]
        require(len(matches) == 1 and matches[0]["command"] == record["compile_command"],
                f"Derived compile command differs: {name}")
        require("CMakeFiles/ncnn.dir/" in matches[0]["command"], "Derived unit is outside ncnn target")
        # These unchanged v2 copies are also present in the final v3 build.
        # Verify the actual derived source, not only its frozen digest record.
        require(identity(matches[0]["file"]) == {key: record[key] for key in ("sha256", "bytes")},
                f"Actual derived source differs: {name}")
    return derived


def verify_run(base, pin):
    plan = read_json(base / "full-plan.json")
    preflight = read_json(base / "preflight.json")
    progress = read_json(base / "full-progress.json")
    comparison = read_json(base / "full-comparison.json")
    require(identity(base / "full-plan.json")["sha256"] == pin["plan"], "Frozen plan changed")
    require(plan["execution_source_head"] == preflight["source_head"] == pin["source"], "Run source mismatch")
    require(preflight["all_passed"] and preflight["source_files"] == pin["source_files"] and
            preflight["bindings_verified"] == len(plan["bindings"]) == pin["bindings"] and
            preflight["derived_compiled_units"] == pin["derived"], "Preflight identity/count differs")
    require(preflight["plan_sha256"] == progress["plan_sha256"] == comparison["plan_sha256"] == pin["plan"],
            "Plan link differs")
    require(progress["status"] == pin["batch_status"] and progress["completed"] == pin["selected"],
            "Original batch status/selection differs")
    require(plan["current_ncnn"] == NCNN and plan["gates"] == GATES, "Dependency or original gates changed")
    for index, (path, record) in enumerate(plan["bindings"].items(), 1):
        require(identity(path) == record, f"Frozen binding differs: {path}")
        if index % 25 == 0:
            print(base.name, "verified bindings", index, flush=True)
    binary = identity(base / "frozen-bin/ernie-image")
    require(binary == preflight["binary"] and binary["sha256"] == pin["binary"], "Binary identity differs")
    require(identity(base / "compare_full.py")["sha256"] == comparison["comparator_sha256"], "Comparator differs")
    require(identity(base / "run_full.py")["sha256"] == progress["supervisor_sha256"], "Supervisor differs")
    model = Path(plan["model"])
    manifest = read_json(model / "manifest.json")
    require(manifest["schema_version"] == 3, "Model schema differs")
    for digest, size in manifest["objects"].items():
        require(plan["bindings"][str(model / "objects" / digest)] == {"sha256": digest, "bytes": size},
                "Model object is missing from frozen bindings")
    source = verify_sources(base, pin)
    derived = verify_derived(base, pin)
    return {"base": base, "plan": plan, "comparison": comparison, "progress": progress,
            "source": source, "derived": derived,
            "record": {"directory": str(base), "source_head": pin["source"], "plan_sha256": pin["plan"],
                       "binary": binary, "source_files_verified": len(source["files"]),
                       "bindings_verified": len(plan["bindings"]), "model_objects_verified": len(manifest["objects"]),
                       "derived_units_verified": len(derived), "batch_status": progress["status"],
                       "selected_cases": pin["selected"], "preflight": identity(base / "preflight.json"),
                       "comparison": identity(base / "full-comparison.json")}}


def verify_lineage(earlier, later):
    old, new = earlier["source"]["files"], later["source"]["files"]
    business = sorted(name for name in set(old) | set(new) if name.startswith(("src/", "include/", "cli/")))
    require(all(old.get(name) == new.get(name) for name in business), "Business source changed between runs")
    changes = sorted(name for name in set(old) | set(new) if old.get(name) != new.get(name))
    require(set(changes) == {"cmake/Dependencies.cmake", "cmake/ErnieNcnnBf16Sdpa.cmake",
                            "cmake/ErnieNcnnBf16.cmake", "tests/CMakeLists.txt", "tests/test_bf16_gemm.cpp"},
            "Unexpected inventoried source change")
    require(not git("diff", "--name-only", PINS["v2"]["source"], PINS["v3"]["source"], "--", "src", "include", "cli").strip(),
            "Git reports business source changes")
    for name, record in earlier["derived"].items():
        require(record == later["derived"].get(name), f"Previously derived unit changed: {name}")
    require(set(later["derived"]) - set(earlier["derived"]) == {"ncnn-bf16-gemm/gemm_vulkan.cpp"},
            "Unexpected derived-unit change")
    require(earlier["record"]["binary"]["sha256"] != later["record"]["binary"]["sha256"], "Distinct binaries were collapsed")
    shared = set(earlier["plan"]["bindings"]) & set(later["plan"]["bindings"])
    require(all(earlier["plan"]["bindings"][name] == later["plan"]["bindings"][name] for name in shared),
            "Shared frozen inputs changed")
    return {"same_binary": False, "same_source_commit": False, "business_files_unchanged": len(business),
            "inventoried_changes": changes, "shared_identical_bindings": len(shared),
            "unchanged_derived_units": len(earlier["derived"]), "added_derived_unit": "ncnn-bf16-gemm/gemm_vulkan.cpp",
            "scope": "All inventoried src/include/cli files match; remaining inventoried changes are BF16 build guards and tests. Documentation/evidence is outside the source inventory."}


def command_options(command):
    result = {}
    index = 1
    while index < len(command):
        flag = command[index]
        require(flag.startswith("--") and flag not in result, "Malformed or duplicate command option")
        if flag == "--text-down-vector":
            result[flag] = True
            index += 1
        else:
            require(index + 1 < len(command), "Missing command value")
            result[flag] = command[index + 1]
            index += 2
    return result


def verify_execution(run, case):
    base, plan = run["base"], run["plan"]
    out = base / "full" / case
    report = read_json(out / "generation.json")
    process = read_json(out / "process.json")
    worker = read_json(out / "worker-result.json")
    command = read_json(out / "command.json")
    require(process["case"] == case and process["complete"] and process["return_code"] == worker["return_code"] == 0,
            "Selected execution did not complete")
    require(process["command"] == worker["command"] == command and command[0] == str(base / "frozen-bin/ernie-image"),
            "Execution command identity differs")
    precision = case.rsplit("-", 1)[1]
    request = report["request"]
    for field, value in {"device": "vulkan", "precision": precision, "vae_device": "cpu", "vae_convolution": "direct",
                         "text_device": "cpu", "text_down_vector": True, "threads": 2, "steps": 8,
                         "dit_weights": "host", "ram_reserve_mib": 3072, "gpu_memory": "auto",
                         "gpu_spill_mib": 2048, "oom_retries": 3, "model_loading": "stdio", "img2img": False}.items():
        require(request[field] == value, f"Request differs: {case}/{field}")
    mixed = case == "mixed-prefetch-fp32"
    require(request["gpu_reserve_mib"] == (5400 if mixed else 512) and
            request["dit_cache_mib"] == request["dit_prefetch_mib"] == (1024 if mixed else 0), "Memory controls differ")
    options = command_options(command)
    if case == "default-fp16":
        require("--precision" not in options and request["precision"] == "fp16", "CLI default FP16 was not exercised")
    else:
        require(options["--precision"] == precision, "Explicit precision differs")
    for field in ("vae_device", "vae_convolution", "threads", "steps", "dit_weights",
                  "gpu_reserve_mib", "dit_cache_mib", "ram_reserve_mib", "model_loading"):
        require(options["--" + field.replace("_", "-")] == str(request[field]), f"Command/report mismatch: {field}")
    for flag, value in {"--model": plan["model"], "--prompt": plan["prompt"], "--latent": plan["initial"],
                        "--output": str(out / "native.png"), "--trace-dir": str(out / "trace"),
                        "--report-json": str(out / "generation.json"), "--width": "512", "--height": "512", "--gpu": "0"}.items():
        require(options[flag] == value, f"Frozen command input differs: {flag}")
    require(options.get("--dit-prefetch-mib", "0") == str(request["dit_prefetch_mib"]), "Prefetch command differs")
    require(report["status"] == "success" and report["shape"] == [512, 512] and report["trace_enabled"] and
            report["token_ids"] == plan["token_ids"] and report["prompt"] == plan["prompt"] and not report["pe"]["enabled"],
            "Generated input/output identity differs")
    require(report["model"] == {"schema_version": 3, "source_width": 1024, "source_height": 1024,
                                "text_bucket": 32, "dit_text_tokens": 64}, "Selected model instance differs")
    require([(p["current"], p["total"]) for p in report["progress"] if p["stage"] == "denoise"] == [(i, 8) for i in range(1, 9)],
            "Not all original denoising steps completed")
    limits = plan["resource_limits"]
    require(limits == {"memory_max_bytes": 17179869184, "host_available_min_bytes": 3221225472,
                       "gpu_whole_device_max_mib": 6144, "timeout_seconds_per_phase": 1800}, "Resource limits changed")
    require(worker["observed_limits"] == {"memory.max": "17179869184", "memory.swap.max": "0", "cpu.max": "200000 100000"},
            "Actual cgroup limits differ")
    events = {key: int(value) for key, value in (line.split() for line in worker["memory_events"].splitlines())}
    require(all(events[key] == 0 for key in ("max", "oom", "oom_kill", "oom_group_kill")), "Resource max/OOM event observed")
    require(worker["memory_peak_bytes"] <= limits["memory_max_bytes"] and
            0 < worker["native_wall_seconds"] <= process["wall_seconds"] <= limits["timeout_seconds_per_phase"] and
            process["minimum_host_available_bytes"] >= limits["host_available_min_bytes"] and
            process["sampled_gpu_whole_device_peak_mib"] <= limits["gpu_whole_device_max_mib"], "Resource outcome exceeds limits")
    identity(out / "resources.jsonl")
    samples = [json.loads(line) for line in (out / "resources.jsonl").read_text().splitlines()]
    require(samples and all(0 <= x["elapsed_seconds"] <= process["wall_seconds"] and
                            x["host_available_bytes"] >= process["minimum_host_available_bytes"] and
                            x["gpu_whole_device_mib"] <= process["sampled_gpu_whole_device_peak_mib"] for x in samples),
            "Raw resource samples disagree with process report")
    errors = validation(out / "native.log")
    require(not any(errors.values()), "Selected execution has Vulkan validation errors")
    memory, prefetch, recovery, cache = (report[key] for key in ("gpu_memory", "weight_prefetch", "memory_recovery", "weight_cache"))
    require(memory["device_allocations"] > 0 and memory["host_allocations"] ==
            memory["host_device_local_allocations"] + memory["host_non_device_local_allocations"] and
            memory["host_peak_bytes"] <= request["gpu_spill_mib"] * 1024**2, "Invalid buffer accounting")
    require(recovery == {"retries": 0, "attention_query_rows": 128}, "Unexpected full-model retry")
    require(prefetch["peak_charged_bytes"] <= request["dit_prefetch_mib"] * 1024**2 and
            cache["peak_charged_bytes"] <= request["dit_cache_mib"] * 1024**2, "Prefetch/cache admission exceeded")
    if mixed:
        require(memory["host_allocations"] == memory["host_non_device_local_allocations"] == 288 and
                memory["host_device_local_allocations"] == 0 and prefetch["started"] == prefetch["used"] == 280 and
                prefetch["skipped"] == 0, "Actual mixed-placement/prefetch mechanism not observed")
    else:
        require(memory["host_allocations"] == 0 and prefetch["started"] == prefetch["used"] == prefetch["skipped"] == 0,
                "Unexpected host buffers or optional prefetch in control")
    return {"case": case, "source_head": plan["execution_source_head"], "binary_sha256": run["record"]["binary"]["sha256"],
            "plan_sha256": identity(base / "full-plan.json")["sha256"], "complete_and_valid": True,
            "process": process, "worker": worker, "resource_samples": len(samples), "memory_events": events,
            "validation": errors, "gpu_memory": memory, "weight_prefetch": prefetch, "weight_cache": cache,
            "memory_recovery": recovery, "generation_report": identity(out / "generation.json"),
            "native_log": identity(out / "native.log"), "resources": identity(out / "resources.jsonl")}


def close(actual, recorded, label):
    require(math.isfinite(actual) and math.isfinite(recorded) and
            math.isclose(actual, recorded, rel_tol=1e-12, abs_tol=1e-15), f"Numerical metric differs: {label}")


def verify_numerics(run, case):
    plan, base = run["plan"], run["base"]
    precision = case.rsplit("-", 1)[1]
    matches = [r for r in run["comparison"]["results"] if r["case"] == case]
    require(len(matches) == 1, "Comparison case missing/duplicated")
    result = matches[0]
    out, reference = base / "full" / case, Path(plan["reference"])
    require(result["process"] == read_json(out / "process.json"), "Comparator process record differs")
    fixture = read_json(reference / "fixture.json")
    require(fixture["complete"] and fixture["steps"] == 8 and fixture["ids"] == plan["token_ids"], "Official fixture differs")
    entries = [(name, value, "conditioning") for name, value in fixture["inputs"].items()]
    for step, values in enumerate(fixture["outputs"]):
        entries.extend((f"{name}-{step}", value, precision) for name, value in values.items())
    entries.extend((name, value, precision) for name, value in fixture["final"].items())
    require(len(entries) == len(result["tensors"]) == 25 and
            {p.name for p in (out / "trace").glob("*.f32")} == {name + ".f32" for name, _, _ in entries}, "Trace membership differs")
    before = Path(plan["saved_" + precision + "_baseline"])
    anchor = read_json(base / "historical-fp32-anchor.json") if precision == "fp32" else None
    if anchor:
        require(hashlib.sha256(git("show", anchor["source_commit"] + ":" + anchor["source_path"])).hexdigest() == anchor["source_sha256"],
                "Historical FP32 anchor source changed")
    historical_fp16 = None
    if precision == "fp16":
        historical_base = before.parents[1]
        historical_plan = read_json(historical_base / "full-plan.json")
        historical_comparison = read_json(historical_base / "full-comparison.json")
        require(historical_comparison["plan_sha256"] == identity(historical_base / "full-plan.json")["sha256"] and
                historical_plan["gates"] == GATES, "Original FP16 gate provenance differs")
        historical_fp16 = next(r for r in historical_comparison["results"] if r["case"] == "candidate-fp16")
        require(historical_fp16["passed_tensors"] == 23 and historical_fp16["png"]["max_abs"] == 109 and
                historical_fp16["passed"] is False, "Historical FP16 negative result differs")
    tensors = []
    for (name, record, gate_name), recorded in zip(entries, result["tensors"]):
        expected_path, actual_path = reference / record["file"], out / "trace" / (name + ".f32")
        require(recorded["name"] == name and record["dtype"] == "float32_le", "Tensor identity/type differs")
        expected_id, actual_id = identity(expected_path), identity(actual_path)
        require(expected_id["sha256"] == record["sha256"] == recorded["reference_sha256"] and
                actual_id["sha256"] == recorded["native_sha256"] and
                actual_id["bytes"] == expected_id["bytes"] == math.prod(record["shape"]) * 4, "Tensor bytes/digest differ")
        expected, actual = np.fromfile(expected_path, "<f4"), np.fromfile(actual_path, "<f4")
        require(np.isfinite(expected).all() and np.isfinite(actual).all() and recorded["finite"], "Non-finite tensor")
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        # Independent sum-of-squares reduction rather than the comparator's
        # np.linalg.norm. Metric agreement is checked; original gates stay exact.
        nrmse = math.sqrt(float(np.sum(difference * difference))) / max(math.sqrt(float(np.sum(expected.astype(np.float64)**2))), 1e-30)
        maximum = float(np.max(np.abs(difference)))
        refmax = float(np.max(np.abs(expected)))
        gate = GATES[gate_name]
        limit = gate["atol"] + gate["global_rtol"] * refmax
        for field, value in {"nrmse": nrmse, "max_abs_error": maximum, "reference_max_abs": refmax,
                             "max_abs_limit": limit, "nrmse_limit": gate["nrmse"]}.items():
            close(value, recorded[field], name + "/" + field)
        passed = nrmse <= gate["nrmse"] and maximum <= limit
        if name == "initial":
            passed = expected_id["sha256"] == actual_id["sha256"]
        require(passed == recorded["passed"] and recorded["elements"] == actual.size, "Original tensor gate differs")
        before_path = before / "trace" / (name + ".f32")
        before_id = identity(before_path)
        if anchor:
            require(before_id["sha256"] == anchor["tensors"][name + ".f32"], "Historical FP32 tensor changed")
        if historical_fp16:
            historical_tensor = next(t for t in historical_fp16["tensors"] if t["name"] == name)
            require(before_id["sha256"] == historical_tensor["native_sha256"], "Historical FP16 tensor changed")
        exact = before_id == actual_id
        old_values = np.fromfile(before_path, "<f4")
        require(old_values.shape == actual.shape and np.isfinite(old_values).all(), "Historical native tensor differs")
        old_difference = float(np.max(np.abs(old_values.astype(np.float64) - actual.astype(np.float64))))
        old_record = next(r for r in result["old_new"]["tensors"] if r["name"] == name)
        require(old_record["bitwise_equal"] == exact, "Native-to-native identity differs")
        close(old_difference, old_record["max_abs_difference"], name + "/old-new")
        tensors.append({"name": name, "elements": int(actual.size), "finite": True, "nrmse": nrmse,
                        "max_abs_error": maximum, "max_abs_limit": limit, "nrmse_limit": gate["nrmse"],
                        "passed": passed, "reference_sha256": expected_id["sha256"], "native_sha256": actual_id["sha256"],
                        "baseline_bitwise_equal": exact, "baseline_max_abs_difference": old_difference})
    png_id = identity(out / "native.png")
    image = np.asarray(Image.open(out / "native.png").convert("RGB"))
    official = np.asarray(Image.open(reference / "reference.png").convert("RGB"))
    require(image.shape == official.shape == (512, 512, 3), "PNG shape differs")
    delta = np.abs(image.astype(np.int16) - official.astype(np.int16))
    mae, maximum = float(delta.mean()), int(delta.max())
    gate = GATES[precision]
    png_passed = mae <= gate["pixel_mae"] and maximum <= gate["pixel_max"]
    decoded = np.fromfile(out / "trace/decoded.f32", "<f4").reshape(3, 512, 512)
    quantized = (np.clip(decoded / 2 + .5, 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    require(np.array_equal(quantized, image), "Decoded-to-PNG quantization differs")
    close(mae, result["png"]["mae"], "PNG MAE")
    require(result["png"]["max_abs"] == maximum and result["png"]["different_channels"] == int(np.count_nonzero(delta)) and
            result["png"]["passed"] == png_passed and result["png"]["sha256"] == png_id["sha256"] and
            result["png"]["quantization_exact"], "PNG comparator result differs")
    before_png = identity(before / "native.png")
    if anchor:
        require(before_png["sha256"] == anchor["png_sha256"], "Historical PNG changed")
    if historical_fp16:
        require(before_png["sha256"] == historical_fp16["png"]["sha256"], "Historical FP16 PNG changed")
    before_image = np.asarray(Image.open(before / "native.png").convert("RGB"))
    png_exact = png_id == before_png
    exact_tensors = sum(t["baseline_bitwise_equal"] for t in tensors)
    require(result["old_new"]["baseline"] == str(before) and result["old_new"]["bitwise_equal_tensors"] == exact_tensors and
            result["old_new"]["png_bitwise_equal"] == png_exact and
            result["old_new"]["png_pixels_equal"] == bool(np.array_equal(image, before_image)) and
            result["old_new"]["png_max_abs_difference"] == int(np.max(np.abs(image.astype(np.int16) - before_image.astype(np.int16)))),
            "Historical PNG/tensor comparison differs")
    passed_count = sum(t["passed"] for t in tensors)
    numerical_pass = passed_count == 25 and png_passed
    require(result["passed_tensors"] == passed_count and result["total_tensors"] == 25 and
            result["total_elements"] == sum(t["elements"] for t in tensors) and result["passed"] == numerical_pass,
            "Aggregate original numerical gate differs")
    if precision == "fp32":
        require(numerical_pass and exact_tensors == 25 and png_exact, "FP32 regression failed")
    return {"original_gates": GATES[precision], "conditioning_gates": GATES["conditioning"],
            "tensors": tensors, "passed_tensors": passed_count, "total_tensors": 25,
            "total_elements": sum(t["elements"] for t in tensors), "official_numerical_pass": numerical_pass,
            "png": {"mae": mae, "max_abs": maximum, "different_channels": int(np.count_nonzero(delta)),
                    "passed": png_passed, "quantization_exact": True, **png_id},
            "baseline": str(before), "baseline_bitwise_equal_tensors": exact_tensors, "baseline_png_bitwise_equal": png_exact}


def main():
    bases = {version: Path("/var/tmp/ernie-memory-execution-20260909-" + version) for version in PINS}
    assert_no_running_models(bases.values())
    runs = {version: verify_run(bases[version], pin) for version, pin in PINS.items()}
    lineage = verify_lineage(runs["v2"], runs["v3"])
    require(runs["v3"]["source"] == runs["v4"]["source"] and runs["v3"]["derived"] == runs["v4"]["derived"] and
            runs["v3"]["record"]["binary"] == runs["v4"]["record"]["binary"], "v3/v4 source or binary differs")
    for field in ("model", "initial", "reference", "prompt", "token_ids", "gates", "resource_limits"):
        require(all(run["plan"][field] == runs["v2"]["plan"][field] for run in runs.values()), f"Shared workload differs: {field}")
    lineage["v3_v4_identical_source_and_binary"] = True
    excluded = bases["v2"] / "full/bf16-flash-bf16"
    excluded_errors = validation(excluded / "native.log")
    require(excluded_errors["vuid_mentions"] > 0 and runs["v2"]["progress"]["current"] == "bf16-flash-bf16",
            "Original invalid BF16 evidence/status was lost")
    excluded_result = next(r for r in runs["v2"]["comparison"]["results"] if r["case"] == "bf16-flash-bf16")
    results = []
    for version, pin in PINS.items():
        for case in pin["selected"]:
            record = verify_execution(runs[version], case)
            record["numerical"] = verify_numerics(runs[version], case)
            results.append(record)
            print(version, case, "valid execution; official", record["numerical"]["passed_tensors"], "/25",
                  "PNG", record["numerical"]["png"]["mae"], record["numerical"]["png"]["max_abs"], flush=True)
    assert_no_running_models(bases.values())
    for path, recorded_stamp in STAMPS.items():
        require(stamp(path) == recorded_stamp, f"Audited input changed before completion: {path}")
    record = {
        "schema_version": 1, "audit_status": "completed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Selected v2 normal/mixed FP32 plus v3 BF16 and v4 actual CLI-default FP16; execution validity and original numerical acceptance are separate",
        "audit_script": identity(__file__), "runs": {key: value["record"] for key, value in runs.items()},
        "source_lineage": lineage, "selected_executions_valid": True,
        "all_selected_official_numerical_pass": all(r["numerical"]["official_numerical_pass"] for r in results),
        "excluded_invalid_execution": {"case": "bf16-flash-bf16", "run": "v2", "accepted": False,
            "reason": "Vulkan cooperative-matrix validation errors; the original batch remains validation_failed",
            "native_log": identity(excluded / "native.log"), "validation": excluded_errors,
            "descriptive_only_official_passed_tensors": excluded_result["passed_tensors"],
            "descriptive_only_png": excluded_result["png"]},
        "results": results, "unique_files_rehashed": len(IDENTITIES),
        "unique_bytes_rehashed": sum(item["bytes"] for item in IDENTITIES.values()),
        "limits": ["Four selected executions use three frozen plans, two source commits and two different binaries; v3/v4 share the final binary.",
            "Mixed placement was triggered by reserved budget, not physical GPU exhaustion.",
            "All selected full models used zero retries; forced checkpoint recovery is a separate small-network test.",
            "FP16/BF16 official numerical failures remain failures despite valid Vulkan execution and any native-to-native equality.",
            "Trace-enabled one-case diagnostics with uncontrolled file cache/background load are not performance acceptance.",
            "Metric recomputation agreement uses 1e-12 relative/1e-15 absolute only to compare independent reductions; the original model and pixel gates are unchanged."],
    }
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("Saved", OUTPUT, "selected execution validity=true; all official numerical gates=",
          record["all_selected_official_numerical_pass"], flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed post-audit for the fixed 1024 positive-strength execution."""
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from reference_img2img_positive import validate_inputs, validate_suffix
except ImportError:
    from tools.reference_img2img_positive import validate_inputs, validate_suffix


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(candidate, reference):
    candidate = np.memmap(candidate, "<f4", "r")
    reference = np.memmap(reference, "<f4", "r")
    if candidate.size != reference.size or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Tensor size or finite contract differs")
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rmse = math.sqrt(float(np.mean(delta * delta)))
    norm = math.sqrt(float(np.mean(np.asarray(reference, dtype=np.float64) ** 2)))
    return {"native_sha256": digest(candidate.filename), "official_sha256": digest(reference.filename),
            "elements": int(candidate.size), "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))), "reference_max_abs": float(np.max(np.abs(reference))),
            "nrmse": rmse / norm if norm else (0. if rmse == 0 else math.inf)}


def validate_identity(base, name):
    identity_path = base / name / "identity.json"
    identity = json.loads(identity_path.read_text())
    root = base / name / "source"
    for relative, expected in identity.get("source_files", {}).items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"{name} source identity differs")
    if name == "native-plan":
        if (digest(base / name / "ernie-image") != identity.get("runner_sha256")
                or digest(Path("outputs/f2-production-img2img-package-1024-v1/manifest.json")) !=
                identity.get("package_manifest_sha256")):
            raise ValueError("Native runner or package identity differs")
    return identity


def validate_process(process):
    if (process.get("complete") is not True or process.get("return_code") != 0 or "failure" in process
            or process.get("cgroup_seen") is not True or process.get("memory_swap_max_observed") != "0"
            or process.get("memory_max_observed") != str(process.get("memory_max_bytes"))
            or process.get("minimum_host_available", 0) < process.get("min_available_bytes", math.inf)
            or process.get("memory_events", "").find("oom 0\n") < 0
            or process.get("memory_events", "").find("oom_kill 0\n") < 0):
        raise ValueError("Execution process or resource guard is incomplete")


def validate_execution_identity(base, execution, plan_identity, contract_sha256,
                                start_sha256=None, noise_sha256=None):
    identity_path = execution / "identity.json"
    identity = json.loads(identity_path.read_text())
    process_path = execution / "process.json"
    process = json.loads(process_path.read_text())
    if (identity.get("input_contract_sha256") != contract_sha256
            or identity.get("start_bitwise_noise") is not True
            or identity.get("start_sha256") != identity.get("noise_sha256")
            or (start_sha256 is not None and identity.get("start_sha256") != start_sha256)
            or (noise_sha256 is not None and identity.get("noise_sha256") != noise_sha256)
            or identity.get("plan_provenance_sha256") != digest(plan_identity)
            or identity.get("process_sha256") != digest(process_path)
            or identity.get("process_command") != process.get("command")):
        raise ValueError("Actual execution identity differs")
    seen = set()
    for item in identity.get("outputs", []):
        relative = item.get("path")
        path = execution / relative if isinstance(relative, str) else execution
        if (not isinstance(relative, str) or relative in seen or not path.is_file()
                or item.get("bytes") != path.stat().st_size or item.get("sha256") != digest(path)):
            raise ValueError("Actual execution output identity differs")
        seen.add(relative)
    if "process.json" not in seen or "runner.log" not in seen:
        raise ValueError("Actual execution identity is incomplete")
    return identity_path


def audit(base):
    base = Path(base); inputs = base / "inputs"; native = base / "native-execution"; trace = native / "trace"
    official_execution = base / "official-execution-v3"
    if not official_execution.exists():
        official_execution = base / "official-execution"
    official = official_execution / "oracle"; suffix = official / "suffix"
    contract, prompt, _ = validate_inputs(inputs)
    contract_sha256 = digest(inputs / "input-contract.json")
    native_identity = validate_identity(base, "native-plan")
    official_identity_path = base / "official-plan/identity-v3.json"
    official_identity = json.loads(official_identity_path.read_text())
    snapshot = base / "official-plan/snapshot"
    for relative, expected in official_identity.get("source_files", {}).items():
        path = snapshot / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Official source identity differs")
    for relative, expected in official_identity.get("snapshot_root_files", {}).items():
        if digest(snapshot / relative) != expected:
            raise ValueError("Official snapshot root identity differs")
    if digest(Path("models/turbo1024-s64-portable/manifest.json")) != official_identity.get("model_manifest_sha256"):
        raise ValueError("Official package identity differs")
    runtime = json.loads((base / "official-plan/runtime-identity.json").read_text())
    for item in runtime.get("sources", {}).values():
        if digest(item["path"]) != item.get("sha256"):
            raise ValueError("Official installed runtime identity differs")
    official_execution_identity = native_execution_identity = None
    if contract["request"]["strength"] == 1.:
        official_execution_identity = validate_execution_identity(
            base, official_execution, official_identity_path, contract_sha256,
            contract["start"]["sha256"], contract["noise"]["sha256"])
        native_execution_identity = validate_execution_identity(
            base, native, base / "native-plan/identity.json", contract_sha256,
            contract["start"]["sha256"], contract["noise"]["sha256"])
    fixture = json.loads((suffix / "fixture.json").read_text())
    start_step = contract["request"]["start_step"]
    suffix_denominator = 9 + 2 * (8 - start_step)
    if validate_suffix(suffix, fixture, prompt, contract["start"]["sha256"], 1024, 1024,
                       start_step) != suffix_denominator:
        raise ValueError("Official suffix denominator differs")
    for process_path in (official_execution / "process.json", native / "process.json"):
        process = json.loads(process_path.read_text())
        validate_process(process)
    if ((trace / "prompt.txt").read_bytes() != (inputs / "prompt.txt").read_bytes()
            or [int(v) for v in (trace / "ids.txt").read_text().split()] != fixture["ids"]
            or (trace / "input.rgb").read_bytes() != (inputs / "input.rgb").read_bytes()):
        raise ValueError("Native input, prompt, or token identity differs")
    pairs = {
        "encoder-mean.f32": (trace / "encoder-mean.f32", official / "out0.f32"),
        "encoder-packed.f32": (trace / "encoder-packed.f32", official / "out1.f32"),
        "encoder-normalized.f32": (trace / "encoder-normalized.f32", official / "out2.f32"),
        "noise.f32": (trace / "noise.f32", official / "saved-noise.f32"),
        "initial.f32": (trace / "initial.f32", suffix / "initial-input.f32"),
        "text.f32": (trace / "text.f32", suffix / "text.f32"),
        "padded-text.f32": (trace / "padded-text.f32", suffix / "padded-text.f32"),
        **{f"constant-{i}.f32": (trace / f"constant-{i}.f32", suffix / f"constant-{i}.f32") for i in range(3)},
        **{f"{kind}-{i}.f32": (trace / f"{kind}-{i}.f32", suffix / f"{kind}-{i}.f32")
           for i in range(start_step, 8) for kind in ("prediction", "step")},
        "final.f32": (trace / "final.f32", suffix / "final.f32"),
        "unpacked.f32": (trace / "unpacked.f32", suffix / "unpacked.f32"),
        "decoded.f32": (trace / "decoded.f32", suffix / "decoded.f32"),
    }
    comparison = {name: metrics(*paths) for name, paths in pairs.items()}
    fp32 = {"nrmse": .003, "global_rtol": .01, "atol": .0002}
    conditioning = {"nrmse": .0002, "global_rtol": .0002, "atol": .0002}
    for name, item in comparison.items():
        gate = conditioning if name.startswith(("text", "padded-text", "constant")) else fp32
        item["gate"] = gate
        item["passed"] = bool(item["nrmse"] <= gate["nrmse"] and item["max_abs"] <=
                              gate["atol"] + gate["global_rtol"] * item["reference_max_abs"])
    candidate = np.asarray(Image.open(native / "output.png").convert("RGB"), dtype=np.int16)
    reference = np.asarray(Image.open(suffix / "reference.png").convert("RGB"), dtype=np.int16)
    if candidate.shape != (1024, 1024, 3) or reference.shape != candidate.shape:
        raise ValueError("PNG shape differs")
    delta = np.abs(candidate - reference)
    comparison["png"] = {"native_sha256": digest(native / "output.png"),
                         "official_sha256": digest(suffix / "reference.png"), "shape": list(candidate.shape),
                         "max_abs": int(delta.max()), "mean_abs": float(delta.mean()),
                         "different_values": int(np.count_nonzero(delta)),
                         "gate": {"pixel_mae": .1, "pixel_max": 2},
                         "passed": bool(delta.mean() <= .1 and delta.max() <= 2)}
    passed = all(item["passed"] for item in comparison.values())
    result = {"schema_version": 1, "status": "pass" if passed else "quality_gate_failed",
              "complete_execution": True, "quality_gate_passed": passed,
              "scope": f"One fixed public development 1024x1024 strength-{contract['request']['strength']} case; not formal15/72",
              "input_contract_sha256": contract_sha256,
              "official_identity_sha256": digest(official_identity_path),
              "official_runtime_identity_sha256": digest(base / "official-plan/runtime-identity.json"),
              "official_process_sha256": digest(official_execution / "process.json"),
              "official_reference_sha256": digest(official / "reference.json"),
              "official_suffix_fixture_sha256": digest(suffix / "fixture.json"),
              "native_identity_sha256": digest(base / "native-plan/identity.json"),
              "native_process_sha256": digest(native / "process.json"),
              "boundaries_compared": len(pairs), "official_suffix_denominator": suffix_denominator,
              "prompt_bytes_equal": True, "token_ids": fixture["ids"]}
    if official_execution_identity is not None:
        result["official_execution_identity_sha256"] = digest(official_execution_identity)
        result["native_execution_identity_sha256"] = digest(native_execution_identity)
    return comparison, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    comparison, result = audit(args.base)
    (args.base / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    result["comparison_sha256"] = digest(args.base / "comparison.json")
    (args.base / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["quality_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

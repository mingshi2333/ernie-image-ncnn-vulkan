#!/usr/bin/env python3
"""Prepare or run the bounded official 1024x1024 VAE encoder reference."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from reference_vae_encoder_large import run_reference as run_encoder_reference
    from package_model import ROOT, sha256
except ImportError:
    from tools.reference_vae_encoder_large import run_reference as run_encoder_reference
    from tools.package_model import ROOT, sha256


WIDTH = HEIGHT = 1024


def development_rgb():
    y, x, c = np.indices((HEIGHT, WIDTH, 3), dtype=np.int32)
    return ((13 * x + 29 * y + 71 * c + (x * y) % 19) % 256).astype(np.uint8)


def resource_plan():
    # The encoder attention position count grows from 4096 at 512x384 to
    # 16384. A dense FP32 positions^2 tensor alone is exactly 1 GiB; common
    # score/probability/workspace coexistence makes extrapolation from RSS
    # unsafe, so the actual worker remains separately guarded and measured.
    positions = 128 * 128
    return {"input": [1, 3, 1024, 1024], "mean": [1, 32, 128, 128],
            "packed": [1, 128, 64, 64], "attention_positions": positions,
            "one_dense_fp32_attention_bytes": positions * positions * 4,
            "recommended_memory_max_bytes": 16 * 1024**3,
            "threads": 2, "swap_max_bytes": 0,
            "status": "estimate_only_run_requires_root_resource_release"}


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new preparation directory")
    output.mkdir(parents=True)
    rgb = development_rgb()
    (output / "input.rgb").write_bytes(rgb.tobytes())
    manifest = {"schema_version": 1, "scope": "1024x1024 development encoder input only; not formal corpus or model evidence",
                "complete": True, "width": WIDTH, "height": HEIGHT,
                "rgb": {"file": "input.rgb", "shape": [HEIGHT, WIDTH, 3], "layout": "HWC_RGB",
                        "sha256": sha256(output / "input.rgb")}, "resource_plan": resource_plan()}
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_reference_1024(prepared, output):
    prepared, output = Path(prepared), Path(output)
    if output.exists():
        raise ValueError("Use a new reference directory")
    prep = json.loads((prepared / "preparation.json").read_text())
    rgb_path = prepared / prep["rgb"]["file"]
    if (prep.get("width"), prep.get("height"), prep["rgb"].get("shape"), prep["rgb"].get("layout")) != (
            WIDTH, HEIGHT, [HEIGHT, WIDTH, 3], "HWC_RGB") or sha256(rgb_path) != prep["rgb"].get("sha256"):
        raise ValueError("Prepared 1024 development input differs")
    rgb = np.fromfile(rgb_path, np.uint8).reshape(HEIGHT, WIDTH, 3)
    fixture = run_encoder_reference(output, WIDTH, HEIGHT, rgb,
                                    "Official 1024x1024 encoder development fixture only; no denoising or formal acceptance")
    fixture["resource_plan"] = resource_plan()
    (Path(output) / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    value = run_reference_1024(args.prepared, args.output) if args.prepared else prepare(args.output)
    print(json.dumps({"output": str(args.output), "scope": value["scope"]}))


if __name__ == "__main__":
    main()

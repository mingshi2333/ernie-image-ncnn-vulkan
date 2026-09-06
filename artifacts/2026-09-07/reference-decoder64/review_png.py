"""Recompute the original peer's two completed 64x64 decoder PNG comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


PLAN_SHA256 = "a161fa7a274a45394fe641289ea18b7a6eb0be6049d2cfb2ab9bed92830f8a2b"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, expected):
    assert actual.shape == expected.shape == (64, 64, 3)
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    values, counts = np.unique(error, return_counts=True)
    return {
        "shape": list(actual.shape),
        "channel_samples": int(error.size),
        "mae": float(error.mean()),
        "max_abs": int(error.max()),
        "different_channels": int(np.count_nonzero(error)),
        "different_pixels": int(np.count_nonzero(np.any(error != 0, axis=2))),
        "histogram": {str(v): int(n) for v, n in zip(values, counts)},
        "historical_pixel_gate_passed": bool(error.mean() <= .1 and error.max() <= 2),
    }


def review(base):
    assert sha(base / "plan.json") == PLAN_SHA256
    plan = json.loads((base / "plan.json").read_text())
    for name, item in plan["bindings"].items():
        path = Path(name)
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], name
    oracle = Path(plan["oracle"]["png"])
    expected = np.array(Image.open(oracle).convert("RGB"))
    decoded = np.fromfile(plan["oracle"]["decoded"], dtype="<f4").reshape(3, 64, 64)
    assert np.isfinite(decoded).all()
    scaled = np.clip(decoded / 2 + .5, 0, 1).transpose(1, 2, 0) * np.float32(255)
    quantized = scaled.round().astype(np.uint8)
    assert np.array_equal(quantized, expected)
    result = {
        "scope": plan["scope"],
        "plan_sha256": PLAN_SHA256,
        "verified_plan_bindings": len(plan["bindings"]),
        "input_sha256": sha(base / "input.f32"),
        "official_png_sha256": sha(oracle),
        "official_decoded_finite": True,
        "official_png_quantization_exact": True,
        "historical_gate": {
            "origin": "assistant_selected_engineering_threshold_not_official_or_peer_standard",
            "calibration_status": "not_calibrated_for_perceptual_quality_or_cross_backend_variation",
            "pixel_mae": .1,
            "pixel_max": 2,
        },
        "official_round_even_vs_half_up": compare(
            quantized, (scaled + np.float32(.5)).astype(np.uint8)),
        "phases": {},
    }
    images = {}
    for phase in ("cpu", "vulkan"):
        process = json.loads((base / phase / "process.json").read_text())
        commands = json.loads((base / phase / "worker-result.json").read_text())
        assert process["complete"] and process["return_code"] == 0
        assert process["plan_sha256"] == PLAN_SHA256
        assert [c["command"] for c in commands] == plan["commands"][phase]
        assert len(commands) == 1 and all(c["return_code"] == 0 for c in commands)
        images[phase] = np.array(Image.open(base / phase / "image.png").convert("RGB"))
        result["phases"][phase] = {
            "output_sha256": sha(base / phase / "image.png"),
            "versus_official": compare(images[phase], expected),
            "process": process,
        }
    result["cpu_vs_vulkan"] = compare(images["cpu"], images["vulkan"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run.resolve())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({phase: item["versus_official"]
                      for phase, item in result["phases"].items()}))

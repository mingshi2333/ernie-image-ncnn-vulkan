#!/usr/bin/env python3
"""Run the reviewed official 512x384 img2img denoising suffix."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

REVIEWED_ENCODER_FIXTURE = "ee5ce5db3cb9912ff5e824bf062d1376bcf9b2e91e942d06924ecd187a93e1a9"
EXPECTED_SHAPE = (1, 128, 24, 32)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_inputs(input_dir):
    input_dir = Path(input_dir)
    contract = json.loads((input_dir / "input-contract.json").read_text())
    request = contract.get("request", {})
    if (request.get("width"), request.get("height"), request.get("steps"),
            request.get("strength"), request.get("start_step"), request.get("denoise_steps")) != (512, 384, 8, .5, 4, 4):
        raise ValueError("Only the reviewed 512x384 eight-step strength-0.5 contract is supported")
    if request.get("pe") != {"enabled": False} or request.get("text_precision") != "fp32" or request.get("text_reduction") != "vector":
        raise ValueError("Conditioning contract differs")
    fixture = input_dir / contract["encoder_fixture"]["file"]
    if digest(fixture) != REVIEWED_ENCODER_FIXTURE or contract["encoder_fixture"].get("sha256") != REVIEWED_ENCODER_FIXTURE:
        raise ValueError("Official encoder fixture is not reviewed")
    for key in ("noise", "start"):
        item = contract[key]
        path = input_dir / item["file"]
        if item.get("shape") != list(EXPECTED_SHAPE) or item.get("dtype") != "<f4" or digest(path) != item.get("sha256"):
            raise ValueError(f"Invalid saved {key}")
        values = np.fromfile(path, "<f4")
        if values.size != int(np.prod(EXPECTED_SHAPE)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved {key} values")
    prompt = (input_dir / contract["prompt"]["file"]).read_text().rstrip("\n")
    if digest(input_dir / contract["prompt"]["file"]) != contract["prompt"]["sha256"] or prompt != contract["prompt"]["text"]:
        raise ValueError("Prompt identity differs")
    return contract, prompt


def run(package, input_dir, output, device):
    try:
        from validate_pipeline import reference
    except ImportError:
        from tools.validate_pipeline import reference
    contract, prompt = validate_inputs(input_dir)
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    suffix = output / "suffix"
    fixture = reference(Path(package), prompt, suffix, 8, device,
                        Path(input_dir) / contract["start"]["file"], 4)
    for name in ("input.rgb", "out0.f32", "out1.f32", "out2.f32", "saved-noise.f32", "start-4.f32"):
        shutil.copyfile(Path(input_dir) / name, output / name)
    result = {
        "schema_version": 1,
        "scope": "Official encoder, BN eps 1e-4, saved FP32 noise, official text and four-step DiT suffix, decoder BN eps 1e-5",
        "complete": fixture.get("complete") is True,
        "input_contract_sha256": digest(Path(input_dir) / "input-contract.json"),
        "official_encoder_fixture_sha256": REVIEWED_ENCODER_FIXTURE,
        "suffix_fixture_sha256": digest(suffix / "fixture.json"),
        "request": contract["request"],
        "prompt": contract["prompt"],
        "boundaries": {name: digest(output / name) for name in ("input.rgb", "out0.f32", "out1.f32", "out2.f32", "saved-noise.f32", "start-4.f32")},
        "suffix": fixture,
    }
    (output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps({"complete": run(args.model, args.inputs, args.output, args.device)["complete"]}))


if __name__ == "__main__":
    main()

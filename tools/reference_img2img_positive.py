#!/usr/bin/env python3
"""Run the reviewed official 512x384 img2img denoising suffix."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

REVIEWED_ENCODER_FIXTURE = "ee5ce5db3cb9912ff5e824bf062d1376bcf9b2e91e942d06924ecd187a93e1a9"
REVIEWED_SOURCE_MANIFEST = "ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59"
EXPECTED_SHAPE = (1, 128, 24, 32)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_start(encoded, noise, start):
    encoded, noise, start = (np.asarray(value) for value in (encoded, noise, start))
    if any(value.dtype != np.float32 or value.shape != EXPECTED_SHAPE or not np.isfinite(value).all()
           for value in (encoded, noise, start)):
        raise ValueError("Invalid positive-strength start inputs")
    expected = np.float32(.8) * noise + (np.float32(1) - np.float32(.8)) * encoded
    if np.asarray(expected, dtype="<f4").tobytes() != np.asarray(start, dtype="<f4").tobytes():
        raise ValueError("Saved start does not equal the reviewed FP32 mixture")


def validate_inputs(input_dir):
    try:
        from reference_img2img import turbo_sigmas
    except ImportError:
        from tools.reference_img2img import turbo_sigmas
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
    encoder = json.loads(fixture.read_text())
    if (encoder.get("width"), encoder.get("height"), encoder.get("posterior"), encoder.get("packing"),
            encoder.get("encoder_bn"), encoder.get("decoder_inverse_bn_eps")) != (
                512, 384, "mode_first_32_channels_no_sampling", "pixel_unshuffle_2",
                {"eps": .0001, "affine": False}, 1e-5):
        raise ValueError("Official encoder math contract differs")
    files = [("rgb", encoder["rgb"], "input.rgb"),
             ("mean", encoder["expected"]["out0"], "out0.f32"),
             ("packed", encoder["expected"]["out1"], "out1.f32"),
             ("normalized", encoder["expected"]["out2"], "out2.f32")]
    for label, item, name in files:
        path = input_dir / name
        if item.get("file") != name or digest(path) != item.get("sha256"):
            raise ValueError(f"Official encoder {label} bytes differ")
    if (contract["encoder_fixture"].get("normalized_sha256") != encoder["expected"]["out2"]["sha256"]
            or contract["encoder_fixture"].get("encoder_bn_eps") != .0001
            or request.get("sigma") != float(np.float32(.8))
            or contract.get("sigmas_f32") != [float(v) for v in turbo_sigmas(8)]):
        raise ValueError("Recorded encoder or schedule contract differs")
    for key in ("noise", "start"):
        item = contract[key]
        path = input_dir / item["file"]
        if item.get("shape") != list(EXPECTED_SHAPE) or item.get("dtype") != "<f4" or digest(path) != item.get("sha256"):
            raise ValueError(f"Invalid saved {key}")
        values = np.fromfile(path, "<f4")
        if values.size != int(np.prod(EXPECTED_SHAPE)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved {key} values")
    encoded = np.fromfile(input_dir / "out2.f32", "<f4").reshape(EXPECTED_SHAPE)
    noise = np.fromfile(input_dir / contract["noise"]["file"], "<f4").reshape(EXPECTED_SHAPE)
    start = np.fromfile(input_dir / contract["start"]["file"], "<f4").reshape(EXPECTED_SHAPE)
    validate_start(encoded, noise, start)
    prompt = (input_dir / contract["prompt"]["file"]).read_text().rstrip("\n")
    if digest(input_dir / contract["prompt"]["file"]) != contract["prompt"]["sha256"] or prompt != contract["prompt"]["text"]:
        raise ValueError("Prompt identity differs")
    return contract, prompt


def validate_suffix(path, fixture, prompt, start_sha256):
    path = Path(path)
    saved = json.loads((path / "fixture.json").read_text())
    if saved != fixture or fixture.get("complete") is not True or fixture.get("prompt") != prompt:
        raise ValueError("Suffix fixture identity or prompt differs")
    config = {"packed_width": 32, "packed_height": 24, "text_bucket": 2048,
              "dit_text_tokens": 2048, "text_layers": 25, "dit_layers": 36}
    if fixture.get("config") != config or fixture.get("steps") != 8 or fixture.get("start_step") != 4:
        raise ValueError("Suffix execution contract differs")
    ids = fixture.get("ids")
    if not isinstance(ids, list) or not ids or any(type(v) is not int for v in ids):
        raise ValueError("Invalid suffix token IDs")
    latent = [1, 128, 24, 32]
    shapes = {"initial": latent, "text": [1, len(ids), 3072], "padded-text": [1, 2048, 3072],
              "constant-0": [1, 2816, 128], "constant-1": [1, 2816, 128],
              "constant-2": [2816, 2816], "final": latent,
              "unpacked": [1, 32, 48, 64], "decoded": [1, 3, 384, 512]}
    required = [("initial", fixture.get("inputs", {}).get("initial")),
                ("text", fixture.get("inputs", {}).get("text")),
                ("padded-text", fixture.get("inputs", {}).get("padded-text"))]
    required += [(f"constant-{i}", fixture.get("inputs", {}).get(f"constant-{i}")) for i in range(3)]
    outputs = fixture.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 4:
        raise ValueError("Suffix must contain four complete denoising steps")
    for offset, item in enumerate(outputs, 4):
        shapes[f"prediction-{offset}"] = latent
        shapes[f"step-{offset}"] = latent
        required += [(f"prediction-{offset}", item.get("prediction")), (f"step-{offset}", item.get("step"))]
    required += [(name, fixture.get("final", {}).get(name)) for name in ("final", "unpacked", "decoded")]
    if len(required) != 17:
        raise ValueError("Suffix denominator differs")
    seen = set()
    for label, item in required:
        expected_file = label + ".f32"
        if label == "initial":
            expected_file = "initial-input.f32"
        if (not isinstance(item, dict) or item.get("file") != expected_file or expected_file in seen
                or item.get("dtype") != "float32_le" or item.get("shape") != shapes[label]
                or digest(path / expected_file) != item.get("sha256")):
            raise ValueError(f"Invalid suffix boundary {label}")
        values = np.fromfile(path / expected_file, "<f4")
        if values.size != int(np.prod(item.get("shape", []))) or not np.isfinite(values).all():
            raise ValueError(f"Invalid suffix values {label}")
        seen.add(expected_file)
    if fixture["inputs"]["initial"]["sha256"] != start_sha256:
        raise ValueError("Suffix initial is not the reviewed img2img start")
    png = path / "reference.png"
    if digest(png) != fixture.get("reference_png_sha256"):
        raise ValueError("Suffix PNG differs")
    return 17


def run(package, input_dir, output, device):
    try:
        from validate_pipeline import reference
    except ImportError:
        from tools.validate_pipeline import reference
    contract, prompt = validate_inputs(input_dir)
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text())
    if digest(package / "manifest.json") != REVIEWED_SOURCE_MANIFEST or manifest.get("config") != {
            "packed_width": 32, "packed_height": 24, "text_bucket": 2048,
            "dit_text_tokens": 2048, "text_layers": 25, "dit_layers": 36}:
        raise ValueError("Official suffix source package is not reviewed")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    suffix = output / "suffix"
    fixture = reference(package, prompt, suffix, 8, device,
                        Path(input_dir) / contract["start"]["file"], 4)
    denominator = validate_suffix(suffix, fixture, prompt, contract["start"]["sha256"])
    for name in ("input.rgb", "out0.f32", "out1.f32", "out2.f32", "saved-noise.f32", "start-4.f32"):
        shutil.copyfile(Path(input_dir) / name, output / name)
    result = {
        "schema_version": 1,
        "scope": "Official encoder, BN eps 1e-4, saved FP32 noise, official text and four-step DiT suffix, decoder BN eps 1e-5",
        "complete": fixture.get("complete") is True,
        "input_contract_sha256": digest(Path(input_dir) / "input-contract.json"),
        "official_encoder_fixture_sha256": REVIEWED_ENCODER_FIXTURE,
        "suffix_fixture_sha256": digest(suffix / "fixture.json"),
        "suffix_tensor_denominator": denominator,
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

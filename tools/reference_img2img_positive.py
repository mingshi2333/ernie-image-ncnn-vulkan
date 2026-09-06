#!/usr/bin/env python3
"""Run the reviewed official 512x384 img2img denoising suffix."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

PROFILES = {
    (512, 384): {
        "encoder_fixture": "ee5ce5db3cb9912ff5e824bf062d1376bcf9b2e91e942d06924ecd187a93e1a9",
        "source_manifest": "ef98859ac741f6923680fb02de39e663fa3fa01943eff9d2d85c6ddaf40c9e59",
        "latent_shape": (1, 128, 24, 32), "text_bucket": 2048,
    },
    (1024, 1024): {
        "encoder_fixture": "88f2e8b7ad63a47fd993b069282dcb8bca9042cf56a470efa84237b711d9f7d9",
        "source_manifest": "72bb195a2d0b3ef2a25f873666f51f4bbec4b391744518597be87206a551efc1",
        "latent_shape": (1, 128, 64, 64), "text_bucket": 64,
    },
}


def reviewed_profile(width, height):
    try:
        return PROFILES[(width, height)]
    except KeyError as error:
        raise ValueError("No reviewed positive-strength profile for this shape") from error


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_inputs(encoder_dir, prompt_file, output, seed=20260906, input_image=None, strength=.5):
    """Freeze inputs without loading text, DiT, or decoder weights."""
    import torch
    from PIL import Image
    encoder_dir, prompt_file, output = Path(encoder_dir), Path(prompt_file), Path(output)
    fixture_path = encoder_dir / "fixture.json"
    fixture = json.loads(fixture_path.read_text())
    profile = reviewed_profile(fixture.get("width"), fixture.get("height"))
    if digest(fixture_path) != profile["encoder_fixture"]:
        raise ValueError("Official encoder fixture is not reviewed")
    if output.exists():
        raise ValueError("Use a new output directory")
    prompt_bytes = prompt_file.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    output.mkdir(parents=True)
    shutil.copyfile(fixture_path, output / "encoder-fixture.json")
    for name in ("input.rgb", "out0.f32", "out1.f32", "out2.f32"):
        shutil.copyfile(encoder_dir / name, output / name)
    if input_image is not None:
        shutil.copyfile(input_image, output / "input.png")
        with Image.open(output / "input.png") as image:
            decoded_rgb = image.convert("RGB").tobytes()
        if (output / "input.rgb").read_bytes() != decoded_rgb:
            raise ValueError("Input image does not decode to the reviewed RGB bytes")
    (output / "prompt.txt").write_bytes(prompt_bytes)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(profile["latent_shape"], generator=generator, dtype=torch.float32).numpy()
    encoded = np.fromfile(output / "out2.f32", "<f4").reshape(profile["latent_shape"])
    if strength not in (.5, 1.):
        raise ValueError("Only the reviewed strength-0.5 and strength-1 endpoints are supported")
    start_step = 4 if strength == .5 else 0
    sigma = np.float32(.8) if strength == .5 else np.float32(1)
    start = (sigma * noise + (np.float32(1) - sigma) * encoded
             if strength == .5 else noise.copy())
    np.asarray(noise, dtype="<f4").tofile(output / "saved-noise.f32")
    start_name = f"start-{start_step}.f32"
    np.asarray(start, dtype="<f4").tofile(output / start_name)
    request = {"width": fixture["width"], "height": fixture["height"], "steps": 8,
               "strength": strength, "start_step": start_step, "denoise_steps": 8 - start_step,
               "sigma": float(sigma), "pe": {"enabled": False},
               "text_precision": "fp32", "text_reduction": "vector"}
    contract = {
        "schema_version": 1, "request": request,
        "encoder_fixture": {"file": "encoder-fixture.json", "sha256": profile["encoder_fixture"],
                            "normalized_sha256": fixture["expected"]["out2"]["sha256"],
                            "encoder_bn_eps": .0001},
        "prompt": {"file": "prompt.txt", "sha256": digest(output / "prompt.txt"), "text": prompt},
        "noise": {"file": "saved-noise.f32", "sha256": digest(output / "saved-noise.f32"),
                  "shape": list(profile["latent_shape"]), "dtype": "<f4",
                  "producer": {"library": "torch", "version": torch.__version__,
                               "algorithm": "torch.randn CPU float32", "seed": seed}},
        "start": {"file": start_name, "sha256": digest(output / start_name),
                  "shape": list(profile["latent_shape"]), "dtype": "<f4",
                  "formula": "float32(sigma*noise + (1-sigma)*official_encoder_normalized)"},
        "sigmas_f32": [1., .9655172228813171, .9230769276618958, .8695651888847351,
                       .800000011920929, .7058823704719543, .5714285969734192,
                       .3636363744735718, 0.],
    }
    if input_image is not None:
        contract["input_image"] = {"file": "input.png", "sha256": digest(output / "input.png"),
                                   "decoded_rgb_sha256": digest(output / "input.rgb")}
    (output / "input-contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    validate_inputs(output)
    return contract


def validate_start(encoded, noise, start, expected_shape=None, strength=.5):
    encoded, noise, start = (np.asarray(value) for value in (encoded, noise, start))
    expected_shape = expected_shape or encoded.shape
    if any(value.dtype != np.float32 or value.shape != tuple(expected_shape) or not np.isfinite(value).all()
           for value in (encoded, noise, start)):
        raise ValueError("Invalid positive-strength start inputs")
    if strength == .5:
        expected = np.float32(.8) * noise + (np.float32(1) - np.float32(.8)) * encoded
    elif strength == 1.:
        expected = noise
    else:
        raise ValueError("Unreviewed positive strength")
    if np.asarray(expected, dtype="<f4").tobytes() != np.asarray(start, dtype="<f4").tobytes():
        raise ValueError("Saved start does not equal the reviewed FP32 endpoint")


def validate_inputs(input_dir):
    try:
        from reference_img2img import turbo_sigmas
    except ImportError:
        from tools.reference_img2img import turbo_sigmas
    input_dir = Path(input_dir)
    contract = json.loads((input_dir / "input-contract.json").read_text())
    request = contract.get("request", {})
    strength = request.get("strength")
    plan = {0.5: (4, 4, float(np.float32(.8))), 1.0: (0, 8, 1.0)}.get(strength)
    if (request.get("steps") != 8 or plan is None
            or (request.get("start_step"), request.get("denoise_steps"), request.get("sigma")) != plan):
        raise ValueError("Only reviewed eight-step strength-0.5/1 contracts are supported")
    profile = reviewed_profile(request.get("width"), request.get("height"))
    if request.get("pe") != {"enabled": False} or request.get("text_precision") != "fp32" or request.get("text_reduction") != "vector":
        raise ValueError("Conditioning contract differs")
    if (contract.get("noise", {}).get("file") != "saved-noise.f32"
            or contract.get("start", {}).get("file") != f"start-{plan[0]}.f32"
            or contract.get("prompt", {}).get("file") != "prompt.txt"):
        raise ValueError("Positive-strength input filenames are not canonical")
    fixture = input_dir / contract["encoder_fixture"]["file"]
    if digest(fixture) != profile["encoder_fixture"] or contract["encoder_fixture"].get("sha256") != profile["encoder_fixture"]:
        raise ValueError("Official encoder fixture is not reviewed")
    encoder = json.loads(fixture.read_text())
    if (encoder.get("width"), encoder.get("height"), encoder.get("posterior"), encoder.get("packing"),
            encoder.get("encoder_bn"), encoder.get("decoder_inverse_bn_eps")) != (
                request["width"], request["height"], "mode_first_32_channels_no_sampling", "pixel_unshuffle_2",
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
            or request.get("sigma") != plan[2]
            or contract.get("sigmas_f32") != [float(v) for v in turbo_sigmas(8)]):
        raise ValueError("Recorded encoder or schedule contract differs")
    for key in ("noise", "start"):
        item = contract[key]
        path = input_dir / item["file"]
        if item.get("shape") != list(profile["latent_shape"]) or item.get("dtype") != "<f4" or digest(path) != item.get("sha256"):
            raise ValueError(f"Invalid saved {key}")
        values = np.fromfile(path, "<f4")
        if values.size != int(np.prod(profile["latent_shape"])) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved {key} values")
    encoded = np.fromfile(input_dir / "out2.f32", "<f4").reshape(profile["latent_shape"])
    noise = np.fromfile(input_dir / contract["noise"]["file"], "<f4").reshape(profile["latent_shape"])
    start = np.fromfile(input_dir / contract["start"]["file"], "<f4").reshape(profile["latent_shape"])
    validate_start(encoded, noise, start, profile["latent_shape"], strength)
    prompt = (input_dir / contract["prompt"]["file"]).read_text()
    if digest(input_dir / contract["prompt"]["file"]) != contract["prompt"]["sha256"] or prompt != contract["prompt"]["text"]:
        raise ValueError("Prompt identity differs")
    if "input_image" in contract:
        from PIL import Image
        item = contract["input_image"]
        if item.get("file") != "input.png" or digest(input_dir / "input.png") != item.get("sha256"):
            raise ValueError("Input image identity differs")
        with Image.open(input_dir / "input.png") as image:
            decoded = image.convert("RGB").tobytes()
        if hashlib.sha256(decoded).hexdigest() != item.get("decoded_rgb_sha256") or decoded != (input_dir / "input.rgb").read_bytes():
            raise ValueError("Input image decoded RGB differs")
    return contract, prompt, profile


def validate_suffix(path, fixture, prompt, start_sha256, width=512, height=384, start_step=4):
    path = Path(path)
    saved = json.loads((path / "fixture.json").read_text())
    if saved != fixture or fixture.get("complete") is not True or fixture.get("prompt") != prompt:
        raise ValueError("Suffix fixture identity or prompt differs")
    profile = reviewed_profile(width, height)
    packed_width, packed_height = width // 16, height // 16
    config = {"packed_width": packed_width, "packed_height": packed_height,
              "text_bucket": profile["text_bucket"], "dit_text_tokens": profile["text_bucket"],
              "text_layers": 25, "dit_layers": 36}
    if fixture.get("config") != config or fixture.get("steps") != 8 or fixture.get("start_step") != start_step:
        raise ValueError("Suffix execution contract differs")
    ids = fixture.get("ids")
    if not isinstance(ids, list) or not ids or any(type(v) is not int for v in ids):
        raise ValueError("Invalid suffix token IDs")
    latent = list(profile["latent_shape"])
    sequence = packed_width * packed_height + profile["text_bucket"]
    shapes = {"initial": latent, "text": [1, len(ids), 3072],
              "padded-text": [1, profile["text_bucket"], 3072],
              "constant-0": [1, sequence, 128], "constant-1": [1, sequence, 128],
              "constant-2": [sequence, sequence], "final": latent,
              "unpacked": [1, 32, height // 8, width // 8], "decoded": [1, 3, height, width]}
    required = [("initial", fixture.get("inputs", {}).get("initial")),
                ("text", fixture.get("inputs", {}).get("text")),
                ("padded-text", fixture.get("inputs", {}).get("padded-text"))]
    required += [(f"constant-{i}", fixture.get("inputs", {}).get(f"constant-{i}")) for i in range(3)]
    outputs = fixture.get("outputs")
    denoise_steps = 8 - start_step
    if not isinstance(outputs, list) or len(outputs) != denoise_steps:
        raise ValueError(f"Suffix must contain {denoise_steps} complete denoising steps")
    for offset, item in enumerate(outputs, start_step):
        shapes[f"prediction-{offset}"] = latent
        shapes[f"step-{offset}"] = latent
        required += [(f"prediction-{offset}", item.get("prediction")), (f"step-{offset}", item.get("step"))]
    required += [(name, fixture.get("final", {}).get(name)) for name in ("final", "unpacked", "decoded")]
    denominator = 9 + 2 * denoise_steps
    if len(required) != denominator:
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
    return denominator


def run(package, input_dir, output, device):
    try:
        from validate_pipeline import reference
    except ImportError:
        from tools.validate_pipeline import reference
    contract, prompt, profile = validate_inputs(input_dir)
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text())
    width, height = contract["request"]["width"], contract["request"]["height"]
    expected_config = {"packed_width": width // 16, "packed_height": height // 16,
                       "text_bucket": profile["text_bucket"], "dit_text_tokens": profile["text_bucket"],
                       "text_layers": 25, "dit_layers": 36}
    if digest(package / "manifest.json") != profile["source_manifest"] or manifest.get("config") != expected_config:
        raise ValueError("Official suffix source package is not reviewed")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    suffix = output / "suffix"
    start_step = contract["request"]["start_step"]
    fixture = reference(package, prompt, suffix, 8, device,
                        Path(input_dir) / contract["start"]["file"], start_step)
    denominator = validate_suffix(suffix, fixture, prompt, contract["start"]["sha256"], width, height, start_step)
    for name in ("input.rgb", "out0.f32", "out1.f32", "out2.f32", "saved-noise.f32", contract["start"]["file"]):
        shutil.copyfile(Path(input_dir) / name, output / name)
    result = {
        "schema_version": 1,
        "scope": f"Official encoder, BN eps 1e-4, saved FP32 noise, official text and {8-start_step}-step DiT suffix, decoder BN eps 1e-5",
        "complete": fixture.get("complete") is True,
        "input_contract_sha256": digest(Path(input_dir) / "input-contract.json"),
        "official_encoder_fixture_sha256": profile["encoder_fixture"],
        "suffix_fixture_sha256": digest(suffix / "fixture.json"),
        "suffix_tensor_denominator": denominator,
        "request": contract["request"],
        "prompt": contract["prompt"],
        "boundaries": {name: digest(output / name) for name in ("input.rgb", "out0.f32", "out1.f32", "out2.f32", "saved-noise.f32", contract["start"]["file"])},
        "suffix": fixture,
    }
    (output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prepare-encoder", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--strength", type=float, choices=(.5, 1.), default=.5)
    args = parser.parse_args()
    if args.prepare_encoder:
        if args.model or args.inputs or not args.prompt_file:
            parser.error("Preparation requires --prepare-encoder and --prompt-file only")
        result = prepare_inputs(args.prepare_encoder, args.prompt_file, args.output, args.seed, args.input_image, args.strength)
        print(json.dumps({"complete": True, "request": result["request"]}))
        return
    if not args.model or not args.inputs or args.prompt_file or args.input_image:
        parser.error("Reference execution requires --model and --inputs")
    print(json.dumps({"complete": run(args.model, args.inputs, args.output, args.device)["complete"]}))


if __name__ == "__main__":
    main()

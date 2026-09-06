#!/usr/bin/env python3
"""Bounded official-module img2img oracle; currently executes reconstruction only."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

try:
    from audit_port_weights import official_inventory, OFFICIAL_REPOSITORY, OFFICIAL_REVISION
    from export_vae_encoder import dimensions, normalize_rgb, rgb_fixture
    from package_model import ROOT, sha256
    from validate_img2img_encoder import bounded_run
except ImportError:
    from tools.audit_port_weights import official_inventory, OFFICIAL_REPOSITORY, OFFICIAL_REVISION
    from tools.export_vae_encoder import dimensions, normalize_rgb, rgb_fixture
    from tools.package_model import ROOT, sha256
    from tools.validate_img2img_encoder import bounded_run


def strength_plan(steps, strength):
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('Steps must be an integer in [1,1000]')
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, (int, float, np.floating)) or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Strength must be finite and in [0,1]')
    value = np.float32(strength)
    count = 0 if value == np.float32(0) else min(
        steps, max(1, int(np.floor(np.float32(steps) * value + np.float32(.5)))))
    return {'denoise_steps': count, 'start_step': steps - count}


def turbo_sigmas(steps):
    strength_plan(steps, 0)
    increment = np.float32(-1) / np.float32(steps)
    values = []
    for i in range(steps):
        # Same endpoint-aware FP32 construction used by the native scheduler.
        raw = (np.float32(1) + increment * np.float32(i) if i < (steps + 1) // 2
               else -increment * np.float32(steps - i))
        values.append(np.float32(4) * raw / (np.float32(1) + np.float32(3) * raw))
    return np.asarray([*values, np.float32(0)], dtype='<f4')


def make_start(encoded, noise, steps, strength):
    encoded = np.asarray(encoded)
    noise = np.asarray(noise)
    if encoded.dtype != np.float32 or encoded.ndim != 4 or encoded.shape[0] != 1 or encoded.shape[1] != 128:
        raise ValueError('Encoded latent must be FP32 NCHW [1,128,H,W]')
    if not np.isfinite(encoded).all():
        raise ValueError('Encoded latent must be finite')
    plan = strength_plan(steps, strength)
    if plan['denoise_steps'] == 0:
        return encoded.copy(), {**plan, 'sigma': 0.0}
    if noise.dtype != np.float32 or noise.shape != encoded.shape or not np.isfinite(noise).all():
        raise ValueError('Saved noise must be finite FP32 with the encoded shape')
    sigma = turbo_sigmas(steps)[plan['start_step']]
    # Preserve the declared FP32 multiplication-then-addition order.
    start = sigma * noise + (np.float32(1) - sigma) * encoded
    return np.asarray(start, dtype='<f4'), {**plan, 'sigma': float(sigma)}


def save_tensor(path, value):
    value = np.asarray(value, dtype='<f4')
    if not np.isfinite(value).all():
        raise ValueError('Cannot save a nonfinite oracle tensor')
    path.write_bytes(value.tobytes())
    return {'file': path.name, 'shape': list(value.shape), 'dtype': '<f4', 'sha256': sha256(path)}


def load_encoder():
    import torch
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file

    official = ROOT / 'models/official'
    config_path = official / 'vae-config.json'
    config_source = json.loads((official / 'vae-config.source.json').read_text())
    config = json.loads(config_path.read_text())
    expected_url = f'{OFFICIAL_REPOSITORY}/resolve/{OFFICIAL_REVISION}/vae/config.json'
    if config_source != {'url': expected_url, 'revision': OFFICIAL_REVISION, 'sha256': sha256(config_path)}:
        raise ValueError('Untrusted VAE config provenance')
    if config.get('latent_channels') != 32 or config.get('patch_size') != [2, 2] or config.get('batch_norm_eps') != 1e-4:
        raise ValueError('Unreviewed VAE encoder configuration')
    selected = ['vae-encoder.safetensors', 'vae-quant.safetensors', 'vae-bn.safetensors']
    official_inventory(official, selected)
    with torch.device('meta'):
        model = AutoencoderKLFlux2.from_config(config)
    manifests = {}
    for label, prefix, module in [('encoder', 'encoder.', model.encoder),
                                  ('quant', 'quant_conv.', model.quant_conv), ('bn', 'bn.', model.bn)]:
        path = official / f'vae-{label}.safetensors'
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        if manifest.get('prefix') != prefix or manifest.get('revision') != OFFICIAL_REVISION:
            raise ValueError('Wrong official encoder component identity')
        state = load_file(path)
        if any(not name.startswith(prefix) for name in state):
            raise ValueError('Foreign official encoder tensor')
        module.load_state_dict({name[len(prefix):]: value.float() for name, value in state.items()},
                               strict=True, assign=True)
        manifests[label] = manifest
    model.eval().requires_grad_(False)
    if model.bn.affine or model.bn.eps != 1e-4:
        raise ValueError('Unexpected encoder BN contract')
    return model, manifests, config_path


def reconstruct(output, width, height, steps, seed):
    dimensions(width, height)
    if output.exists():
        raise ValueError('Use a new oracle output directory')
    output.mkdir(parents=True)
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.ernie_image.pipeline_ernie_image import ErnieImagePipeline

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    encoder, encoder_manifests, config_path = load_encoder()
    rgb = rgb_fixture(width, height)
    (output / 'input.rgb').write_bytes(rgb.tobytes())
    normalized = torch.from_numpy(normalize_rgb(rgb))
    mean = encoder.encode(normalized).latent_dist.mode()
    packed = torch.nn.functional.pixel_unshuffle(mean, 2)
    encoded = encoder.bn(packed)
    noise = torch.randn(encoded.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    start, plan = make_start(encoded.numpy(), noise.numpy(), steps, 0.)
    # Reconstruction deliberately performs no text, scheduler step, or DiT load.
    unpacked = ErnieImagePipeline._unpatchify_latents(
        torch.from_numpy(start) * torch.sqrt(encoder.bn.running_var.reshape(1, 128, 1, 1) + 1e-5)
        + encoder.bn.running_mean.reshape(1, 128, 1, 1))
    try:
        from export_vae import load_vae
    except ImportError:
        from tools.export_vae import load_vae
    decoder, decoder_manifests = load_vae()
    decoded = decoder._decode(unpacked, return_dict=False)[0]
    pixels = ((decoded[0] / 2 + .5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    Image.fromarray(pixels).save(output / 'reconstruction.png')
    from diffusers import __version__ as diffusers_version
    manifest = {
        'schema_version': 1, 'suite': 'official-module-img2img-reference-v1',
        'scope': 'Official encoder and decoder reconstruction; no text or DiT loaded; not a published official img2img pipeline',
        'complete': True, 'official_revision': OFFICIAL_REVISION,
        'source': {'rgb': {'file': 'input.rgb', 'shape': [height, width, 3], 'layout': 'HWC_RGB',
                           'sha256': sha256(output / 'input.rgb'),
                           'normalization': 'FP32 (v-127.5)*(1/127.5)'},
                   'saved_noise_seed': seed},
        'request': {'width': width, 'height': height, 'steps': steps, 'strength': 0., **plan},
        'contracts': {'posterior': 'mode_mean_no_sampling', 'packing': 'pixel_unshuffle_2',
                      'encoder_bn': {'eps': 1e-4, 'affine': False}, 'decoder_inverse_bn_eps': 1e-5,
                      'fp32_operation_order': 'sigma*noise + (1-sigma)*encoded'},
        'weights': {**{f'encoder_{k}': v['sha256'] for k, v in encoder_manifests.items()},
                    **{f'decoder_{k}': v['sha256'] for k, v in decoder_manifests.items()}},
        'sources': {'tool_sha256': sha256(__file__), 'vae_config_sha256': sha256(config_path),
                    'autoencoder_source_sha256': sha256(inspect.getfile(AutoencoderKLFlux2)),
                    'pipeline_source_sha256': sha256(inspect.getfile(ErnieImagePipeline)),
                    'diffusers_version': diffusers_version},
        'tensors': {'normalized_rgb': save_tensor(output / 'normalized-rgb.f32', normalized.numpy()),
                    'mean': save_tensor(output / 'mean.f32', mean.numpy()),
                    'packed': save_tensor(output / 'packed.f32', packed.numpy()),
                    'encoded': save_tensor(output / 'encoded.f32', encoded.numpy()),
                    'saved_noise': save_tensor(output / 'saved-noise.f32', noise.numpy()),
                    'start': save_tensor(output / 'start.f32', start),
                    'unpacked': save_tensor(output / 'unpacked.f32', unpacked.numpy()),
                    'decoded': save_tensor(output / 'decoded.f32', decoded.numpy())},
        'reconstruction_png_sha256': sha256(output / 'reconstruction.png')}
    (output / 'reference.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--strength', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.strength != 0:
        parser.error('Only strength=0 reconstruction is executable before the full DiT oracle is integrated')
    if args.worker:
        reconstruct(args.output, args.width, args.height, args.steps, args.seed)
        return 0
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    oracle = args.output / 'oracle'
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(oracle.resolve()),
               '--width', str(args.width), '--height', str(args.height), '--steps', str(args.steps),
               '--strength', '0', '--seed', str(args.seed)]
    process = bounded_run(command, args.output / 'process')
    result = {'schema_version': 1, 'scope': 'Bounded official strength-zero reconstruction',
              'process': process, 'passed': False}
    if process['passed']:
        reference = json.loads((oracle / 'reference.json').read_text())
        result.update(passed=reference.get('complete') is True,
                      reference_manifest_sha256=sha256(oracle / 'reference.json'),
                      reconstruction_png_sha256=reference['reconstruction_png_sha256'])
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'output': str(args.output)}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
